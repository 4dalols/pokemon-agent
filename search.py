"""Determinized rollout search on top of the heuristic policy.

For a MAIN selection, hidden cards (own deck order and prizes, the opponent's deck,
prizes and hand) are sampled; each candidate option is then played through the native
simulator with the heuristic policy acting for both sides until the opponent's next
turn ends. Candidates are ranked by the averaged outcome; the heuristic ranking is the
fallback whenever no native engine is importable or the time budget is spent.

When the game can end this turn, the exact `LethalSolver` runs first on the same
determinizations and its answer wins over the rollouts. Inside each rollout the same
solver checks whether the opponent has a forced win in their reply, which counts as a
loss for the candidate.
"""

import ctypes
import importlib
import json
import random
import time
from collections import Counter
from typing import cast

from lethal import LethalSolver
from policy import Policy
from schema import Current, Observation, Player, SearchState

TERMINAL = 10_000.0
STEP_LIMIT = 400


def load_engine() -> ctypes.CDLL | None:
    for module in ("kaggle_environments.envs.cabt.cg.sim", "cg.sim"):
        try:
            return declare(cast(ctypes.CDLL, importlib.import_module(module).lib))
        except ImportError:
            continue
    return None


def declare(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Declare the search ABI; the kaggle_environments copy of cg/sim.py omits it."""
    pointer = ctypes.POINTER(ctypes.c_int)
    lib.AgentStart.restype = ctypes.c_void_p
    lib.AgentStart.argtypes = []
    lib.SearchBegin.restype = ctypes.c_char_p
    lib.SearchBegin.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
    ]
    lib.SearchStep.restype = ctypes.c_char_p
    lib.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_int64, pointer, ctypes.c_int]
    lib.SearchEnd.restype = None
    lib.SearchEnd.argtypes = [ctypes.c_void_p]
    lib.SearchRelease.restype = None
    lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    return lib


class Searcher:
    def __init__(
        self,
        policy: Policy,
        engine: ctypes.CDLL | None,
        budget: float = 1.5,
        candidates: int = 6,
        seed: int | None = None,
        lethal_budget: float = 0.5,
        lethal_determinizations: int = 4,
        threat_check: bool = True,
    ) -> None:
        self.policy = policy
        self.engine = engine
        self.budget = budget
        self.candidates = candidates
        self.rng = random.Random(seed)
        self.agent = engine.AgentStart() if engine is not None else None
        self.calls = 0
        self.samples = 0
        self.lethal_budget = lethal_budget
        self.lethal_determinizations = lethal_determinizations
        self.threat_check = threat_check
        self.solver = LethalSolver(policy, self, determinizations=lethal_determinizations)
        self.deck_pool = Counter(policy.deck)
        self.basic_energy = {
            card["energyType"]: card["cardId"]
            for card in sorted(policy.cards.values(), key=lambda card: card["cardId"])
            if card["cardType"] == 5 and card["name"].startswith("Basic ")
        }
        self.basics = [
            card["cardId"]
            for card in policy.cards.values()
            if card["basic"] and card["cardType"] in (1, 2, 3, 4)
        ]

    def choose(self, observation: Observation, remaining: float | None = None) -> list[int]:
        fallback = self.policy.choose(observation)
        selection, current = observation["select"], observation["current"]
        serialized = observation.get("search_begin_input")
        if (
            self.agent is None
            or selection is None
            or current is None
            or not serialized
            or selection["type"] != 0
            or len(selection["option"]) < 2
            or current["turn"] < 2
            or any(card is None for card in self.opponent(current)["active"])
        ):
            return fallback
        budget = self.budget if remaining is None else min(self.budget, (remaining - 100) / 250)
        if budget <= 0.02:
            return fallback
        started = time.perf_counter()
        scores = self.policy.scores(selection, current)
        if self.lethal_budget > 0 and self.solver.can_finish(current, current["yourIndex"]):
            deadline = started + min(budget / 2, self.lethal_budget)
            lethal = self.lethal(current, serialized, scores, deadline)
            if lethal is not None:
                return [lethal]
        ranked = self.policy.rank(scores)[: self.candidates]
        if fallback[0] not in ranked:
            ranked.insert(0, fallback[0])
        return [self.search(current, serialized, ranked, started + budget)]

    def lethal(
        self, current: Current, serialized: str, scores: list[float], deadline: float
    ) -> int | None:
        first = [index for index in self.policy.rank(scores) if scores[index] > -100]
        try:
            roots: list[SearchState] = []
            for _ in range(self.lethal_determinizations):
                root = self.begin(current, serialized, manual_coin=True)
                if root is None:
                    break
                roots.append(root)
            if not roots or not first:
                return None
            hit = self.solver.solve(roots, first, deadline)
        finally:
            self.lib.SearchEnd(self.agent)
        return None if hit is None else hit[0]

    def search(self, current: Current, serialized: str, ranked: list[int], deadline: float) -> int:
        me = current["yourIndex"]
        totals = {index: 0.0 for index in ranked}
        samples = 0
        try:
            while time.perf_counter() < deadline:
                root = self.begin(current, serialized)
                if root is None:
                    break
                for index in ranked:
                    totals[index] += self.rollout(root, [index], me, current["turn"], deadline)
                samples += 1
        finally:
            self.lib.SearchEnd(self.agent)
        self.calls += 1
        self.samples += samples
        if samples == 0:
            return ranked[0]
        return max(ranked, key=lambda index: (totals[index], -ranked.index(index)))

    def begin(
        self, current: Current, serialized: str, manual_coin: bool = False
    ) -> SearchState | None:
        me = current["yourIndex"]
        mine, theirs = current["players"][me], current["players"][1 - me]
        own_unseen = list((self.deck_pool - Counter(self.seen(mine))).elements())
        self.rng.shuffle(own_unseen)
        hidden_prizes = sum(card is None for card in mine["prize"])
        if len(own_unseen) < hidden_prizes + mine["deckCount"]:
            return None
        my_prize = own_unseen[:hidden_prizes]
        my_deck = own_unseen[hidden_prizes:]
        enemy = self.enemy_pool(theirs)
        self.rng.shuffle(enemy)
        enemy_prizes = sum(card is None for card in theirs["prize"])
        enemy_hand = theirs["handCount"]
        payload = serialized.encode("ascii")
        raw = self.lib.SearchBegin(
            self.agent,
            payload,
            len(payload),
            ints(my_deck),
            ints(my_prize),
            ints(enemy[enemy_prizes + enemy_hand :]),
            ints(enemy[:enemy_prizes]),
            ints(enemy[enemy_prizes : enemy_prizes + enemy_hand]),
            ints([]),
            int(manual_coin),
        )
        result = json.loads(raw)
        if result["error"] != 0:
            return None
        return cast(SearchState, result["state"])

    def rollout(
        self, root: SearchState, first: list[int], me: int, start_turn: int, deadline: float
    ) -> float:
        state = self.step(root["searchId"], first)
        if state is None:
            return -TERMINAL
        observation = state["observation"]
        checked = False
        for _ in range(STEP_LIMIT):
            current = observation["current"]
            selection = observation["select"]
            if current is None or selection is None or current["result"] >= 0:
                break
            if current["yourIndex"] == me and current["turn"] >= start_turn + 2:
                break
            if (
                self.threat_check
                and not checked
                and current["yourIndex"] != me
                and selection["type"] == 0
                and self.solver.can_finish(current, 1 - me)
            ):
                checked = True
                threat = self.solver.threat(state, min(deadline, time.perf_counter() + 0.05))
                if threat > 0:
                    return -TERMINAL * threat
            state = self.step(state["searchId"], self.policy.choose(observation))
            if state is None:
                return -TERMINAL
            observation = state["observation"]
        return self.evaluate(observation, me)

    def step(self, search_id: int, select: list[int]) -> SearchState | None:
        raw = self.lib.SearchStep(self.agent, search_id, ints(select), len(select))
        result = json.loads(raw)
        if result["error"] != 0:
            return None
        return cast(SearchState, result["state"])

    def release(self, search_id: int) -> None:
        self.lib.SearchRelease(self.agent, search_id)

    def evaluate(self, observation: Observation, me: int) -> float:
        current = observation["current"]
        if current is None:
            return 0.0
        if current["result"] >= 0:
            if current["result"] == me:
                return TERMINAL
            return -TERMINAL if current["result"] == 1 - me else 0.0
        mine, theirs = current["players"][me], current["players"][1 - me]
        return (
            300.0 * (len(theirs["prize"]) - len(mine["prize"]))
            + self.board(theirs) * -1
            + self.board(mine)
        )

    @staticmethod
    def board(player: Player) -> float:
        score = 0.0
        for card in player["active"] + player["bench"]:
            if card is None:
                continue
            score += 40 + 25 * len(card.get("energies", []))
            score -= 0.6 * (card.get("maxHp", 0) - card.get("hp", 0))
        return score

    def enemy_pool(self, theirs: Player) -> list[int]:
        seen = Counter(self.seen(theirs))
        needed = theirs["deckCount"] + theirs["handCount"]
        needed += sum(card is None for card in theirs["prize"])
        if seen <= self.deck_pool:
            mirror = list((self.deck_pool - seen).elements())
            self.rng.shuffle(mirror)
            if len(mirror) >= needed:
                return mirror[:needed]
        pool: list[int] = []
        for card_id, count in seen.items():
            data = self.policy.cards.get(card_id)
            copies = 60 if data is not None and data["cardType"] == 5 else max(0, 4 - count)
            pool.extend([card_id] * min(copies, 4))
        types = Counter(
            self.policy.cards[card["id"]]["energyType"]
            for card in theirs["active"] + theirs["bench"]
            if card is not None and card["id"] in self.policy.cards
        )
        energy_type = types.most_common(1)[0][0] if types else 11
        filler = self.basic_energy.get(energy_type, self.basic_energy.get(11, 3))
        self.rng.shuffle(pool)
        pool = pool[:needed]
        pool.extend([filler] * (needed - len(pool)))
        return pool

    @staticmethod
    def seen(player: Player) -> list[int]:
        ids = [card["id"] for card in player["hand"] or []]
        ids.extend(card["id"] for card in player["discard"])
        ids.extend(card["id"] for card in player["prize"] if card is not None)
        for card in player["active"] + player["bench"]:
            if card is None:
                continue
            ids.append(card["id"])
            for attached in card.get("energyCards", []) + card.get("tools", []):
                ids.append(attached["id"])
            ids.extend(pre["id"] for pre in card.get("preEvolution", []))
        return ids

    @staticmethod
    def opponent(current: Current) -> Player:
        return current["players"][1 - current["yourIndex"]]

    @property
    def lib(self) -> ctypes.CDLL:
        if self.engine is None:
            raise RuntimeError("No native engine available")
        return self.engine


def ints(values: list[int]) -> ctypes.Array[ctypes.c_int]:
    return (ctypes.c_int * len(values))(*values)
