"""Determinized rollout search on top of the heuristic policy.

At each searchable prompt the hidden cards (own deck order and prizes, the opponent's
deck, prizes and hand) are sampled; each candidate answer is then played through the
native simulator until the opponent's next turn ends, our heuristic policy acting for
our side and a heuristic built for the opponent's observed plus sampled deck acting for
theirs. Candidates are ranked by the averaged outcome (terminal results, otherwise the
learned win probability from `value.py` when weights are loaded, otherwise the
hand-written prize/board score), with successive halving dropping the weakest half of
the field at fixed fractions of the time budget. The heuristic answer is the fallback
whenever no native engine is importable or the time budget is spent.
"""

import ctypes
import importlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import replace
from typing import TypedDict, cast

from archetypes import Predictor, Sampler
from memory import Threat, Tracker
from policy import Policy
from schema import Current, Observation, Player, Selection
from value import Featurizer, ValueModel

TERMINAL = 10_000.0
VALUE_SCALE = 1_000.0
STEP_LIMIT = 400
RIVAL_CACHE = 256
MAIN, ENERGY = 0, 4
END = 14
LIKELY = 0.5


class SearchState(TypedDict):
    observation: Observation
    searchId: int


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
        candidates: int = 8,
        seed: int | None = None,
        model: ValueModel | None = None,
        prompts: str = "all",
        halving: bool = True,
        reserve: float = 60.0,
        per_game: int = 40,
        games: int = 3,
        horizon: int = 1,
        epsilon: float = 0.0,
        tracker: Tracker | None = None,
        idle: str = "heuristic",
    ) -> None:
        self.policy = policy
        self.tracker = tracker
        self.engine = engine
        self.model = model
        self.featurizer = Featurizer(policy.cards, policy.attacks)
        self.budget = budget
        self.candidates = candidates
        self.prompts = prompts
        self.halving = halving
        self.reserve = reserve
        self.per_game = per_game
        self.games = games
        self.horizon = horizon
        self.epsilon = epsilon
        self.idle = idle
        self.rng = random.Random(seed)
        self.rival = policy
        self.own_key = tuple(sorted(policy.deck))
        self.rivals: dict[tuple[int, ...], Policy] = {}
        self.agent = engine.AgentStart() if engine is not None else None
        self.calls = 0
        self.samples = 0
        self.spent = 0.0
        self.deck_pool = Counter(policy.deck)
        self.basic_energy = {
            card["energyType"]: card["cardId"]
            for card in sorted(policy.cards.values(), key=lambda card: card["cardId"])
            if card["cardType"] == 5 and card["name"].startswith("Basic ")
        }
        self.predictor = Predictor(policy.deck, policy.cards, self.rng)
        self.prepared_for: Current | None = None
        self.own_fixed: tuple[list[int], list[int]] | None = None
        self.own_unseen: list[int] = []
        self.enemy_sampler: Sampler | None = None
        self.enemy_uniform: list[int] = []
        self.basics = [
            card["cardId"]
            for card in policy.cards.values()
            if card["basic"] and card["cardType"] in (1, 2, 3, 4)
        ]

    def choose(self, observation: Observation, remaining: float | None = None) -> list[int]:
        self.remember(observation)
        fallback = self.policy.choose(observation)
        selection, current = observation["select"], observation["current"]
        serialized = observation.get("search_begin_input")
        if (
            self.agent is None
            or selection is None
            or current is None
            or not serialized
            or selection["type"] == ENERGY
            or (selection["type"] != MAIN and self.prompts != "all")
            or current["turn"] < 2
            or any(card is None for card in self.opponent(current)["active"])
        ):
            return fallback
        candidates = self.candidates_for(selection, current, fallback)
        if len(candidates) < 2:
            return fallback
        budget = self.allocate(current, remaining)
        if budget <= 0.02:
            return fallback
        return self.search(current, serialized, candidates, budget)

    def allocate(self, current: Current, remaining: float | None) -> float:
        """Per-decision seconds: the cap, shrunk so the match stays inside its overage pool."""
        if remaining is None:
            return self.budget
        games_left = max(1, self.games - current.get("round", 1) + 1)
        return min(self.budget, (remaining - self.reserve) / (games_left * self.per_game))

    def candidates_for(
        self, selection: Selection, current: Current, fallback: list[int]
    ) -> list[list[int]]:
        """Distinct legal answers to try, the heuristic answer first."""
        scores = self.policy.scores(selection, current)
        ranked = self.policy.rank(scores)
        if self.idle == "heuristic" and selection["type"] == MAIN:
            ending = {i for i, option in enumerate(selection["option"]) if option["type"] == END}
            if ranked[0] not in ending and any(scores[i] > 0 for i in ranked if i not in ending):
                ranked = [i for i in ranked if i not in ending]
        low, high = selection["minCount"], min(selection["maxCount"], len(ranked))
        candidates = [fallback]
        if high <= 1:
            candidates.extend([index] for index in ranked)
            if low == 0:
                candidates.append([])
        else:
            chosen = set(fallback)
            unchosen = [index for index in ranked if index not in chosen]
            weakest = sorted(fallback, key=lambda index: (scores[index], -index))
            for out in weakest:
                for inside in unchosen:
                    candidates.append([inside if index == out else index for index in fallback])
            if len(fallback) > low:
                candidates.extend([index for index in fallback if index != out] for out in weakest)
            if len(fallback) < high:
                candidates.extend(fallback + [inside] for inside in unchosen)
        distinct: list[list[int]] = []
        for candidate in candidates:
            if low <= len(candidate) <= high and candidate not in distinct:
                distinct.append(candidate)
        return distinct[: self.candidates]

    def search(
        self, current: Current, serialized: str, candidates: list[list[int]], budget: float
    ) -> list[int]:
        me = current["yourIndex"]
        alive = list(range(len(candidates)))
        totals = [0.0] * len(candidates)
        rounds = max(1, math.ceil(math.log2(len(alive)))) if self.halving else 0
        stage = 0
        samples = 0
        started = time.perf_counter()
        try:
            while (elapsed := time.perf_counter() - started) < budget:
                cut = budget * (stage + 1) / (rounds + 1)
                if stage < rounds and len(alive) > 2 and elapsed >= cut:
                    alive.sort(key=lambda index: (-totals[index], index))
                    alive = alive[: max(2, math.ceil(len(alive) / 2))]
                    stage += 1
                root = self.begin(current, serialized)
                if root is None:
                    break
                for index in alive:
                    totals[index] += self.rollout(root, candidates[index], me, current["turn"])
                samples += 1
        finally:
            self.lib.SearchEnd(self.agent)
        self.calls += 1
        self.samples += samples
        self.spent += time.perf_counter() - started
        if samples == 0:
            return candidates[0]
        return candidates[max(alive, key=lambda index: (totals[index], -index))]

    def remember(self, observation: Observation) -> None:
        """Feed the observation to the tracker; a broken log never breaks the agent."""
        if self.tracker is None:
            return
        current = observation["current"]
        try:
            self.tracker.observe(observation)
            if current is not None:
                threat = self.tracker.threat(current, self.policy.cards, self.policy.attacks)
                revealed = self.tracker.revealed(self.opponent(current), current)
                expected = self.expected_pokemon(revealed)
                self.policy.threat = replace(
                    threat, attackers=tuple(sorted(set(threat.attackers) | expected))
                )
        except (KeyError, TypeError, ValueError, AttributeError, IndexError):
            self.tracker.reset()
            self.policy.threat = Threat()

    def expected_pokemon(self, revealed: Counter[int]) -> set[int]:
        """Pokémon the opponent probably runs: posterior mass over the archetype library."""
        posterior = self.predictor.posterior(revealed)
        mass: dict[int, float] = {}
        for entry in self.predictor.library:
            weight = posterior.get(entry.name, 0.0)
            if weight > 0.0:
                for card_id in entry.cards:
                    data = self.policy.cards.get(card_id)
                    if data is not None and data["cardType"] == 0:
                        mass[card_id] = mass.get(card_id, 0.0) + weight
        return {card_id for card_id, weight in mass.items() if weight >= LIKELY}

    def prepare(self, current: Current) -> None:
        """Resolve the tracker/predictor once per decision; rollouts only reshuffle."""
        if self.prepared_for is current:
            return
        me = current["yourIndex"]
        mine, theirs = current["players"][me], current["players"][1 - me]
        self.own_fixed = None
        self.enemy_sampler = None
        if self.tracker is not None:
            self.own_fixed = self.tracker.own_hidden(mine, current)
            side = self.tracker.opponent(current)
            revealed = self.tracker.revealed(theirs, current)
            self.enemy_sampler = self.predictor.prepare(theirs, current, side, revealed)
        self.own_unseen = list((self.deck_pool - Counter(self.seen(mine))).elements())
        self.enemy_uniform = self.enemy_pool(theirs)
        self.prepared_for = current

    def own_hidden(self, mine: Player, current: Current) -> tuple[list[int], list[int]] | None:
        """Our (deck, prizes): exact from the tracker when known, otherwise sampled."""
        self.prepare(current)
        if self.own_fixed is not None:
            deck, prizes = self.own_fixed
            deck = list(deck)
            self.rng.shuffle(deck)
            return deck, list(prizes)
        own_unseen = list(self.own_unseen)
        self.rng.shuffle(own_unseen)
        hidden_prizes = sum(card is None for card in mine["prize"])
        if len(own_unseen) < hidden_prizes + mine["deckCount"]:
            return None
        return own_unseen[hidden_prizes:], own_unseen[:hidden_prizes]

    def enemy_hidden(
        self, theirs: Player, current: Current
    ) -> tuple[list[int], list[int], list[int]]:
        """Opponent (deck, hand, prizes) from the archetype mixture, else the old pool."""
        self.prepare(current)
        if self.enemy_sampler is not None:
            deck, hand, prizes = self.enemy_sampler.sample()
            self.rival = self.rival_policy(self.seen(theirs) + deck + hand + prizes)
            return deck, hand, prizes
        enemy = list(self.enemy_uniform)
        self.rival = self.rival_policy(self.seen(theirs) + enemy)
        self.rng.shuffle(enemy)
        enemy_prizes = sum(card is None for card in theirs["prize"])
        enemy_hand = theirs["handCount"]
        return (
            enemy[enemy_prizes + enemy_hand :],
            enemy[enemy_prizes : enemy_prizes + enemy_hand],
            enemy[:enemy_prizes],
        )

    def begin(self, current: Current, serialized: str) -> int | None:
        me = current["yourIndex"]
        mine, theirs = current["players"][me], current["players"][1 - me]
        own = self.own_hidden(mine, current)
        if own is None:
            return None
        my_deck, my_prize = own
        enemy_deck, enemy_hand, enemy_prize = self.enemy_hidden(theirs, current)
        payload = serialized.encode("ascii")
        raw = self.lib.SearchBegin(
            self.agent,
            payload,
            len(payload),
            ints(my_deck),
            ints(my_prize),
            ints(enemy_deck),
            ints(enemy_prize),
            ints(enemy_hand),
            ints([]),
            0,
        )
        result = json.loads(raw)
        if result["error"] != 0:
            return None
        return int(result["state"]["searchId"])

    def rollout(self, root: int, first: list[int], me: int, start_turn: int) -> float:
        state = self.step(root, first)
        if state is None:
            return -TERMINAL
        observation = state["observation"]
        for _ in range(STEP_LIMIT):
            current = observation["current"]
            selection = observation["select"]
            if current is None or selection is None or current["result"] >= 0:
                break
            if current["yourIndex"] == me and current["turn"] >= start_turn + 2 * self.horizon:
                break
            actor = self.policy if current["yourIndex"] == me else self.rival
            state = self.step(state["searchId"], self.playout_choice(observation, selection, actor))
            if state is None:
                return -TERMINAL
            observation = state["observation"]
        return self.evaluate(observation, me)

    def playout_choice(
        self, observation: Observation, selection: Selection, actor: Policy
    ) -> list[int]:
        """The actor's heuristic answer; with probability epsilon one of its top three options."""
        if (
            self.epsilon > 0.0
            and selection["type"] == MAIN
            and len(selection["option"]) > 1
            and observation["current"] is not None
            and self.rng.random() < self.epsilon
        ):
            ranked = actor.rank(actor.scores(selection, observation["current"]))
            return [self.rng.choice(ranked[:3])]
        return actor.choose(observation)

    def step(self, search_id: int, select: list[int]) -> SearchState | None:
        raw = self.lib.SearchStep(self.agent, search_id, ints(select), len(select))
        result = json.loads(raw)
        if result["error"] != 0:
            return None
        return cast(SearchState, result["state"])

    def evaluate(self, observation: Observation, me: int) -> float:
        current = observation["current"]
        if current is None:
            return 0.0
        if current["result"] >= 0:
            if current["result"] == me:
                return TERMINAL
            return -TERMINAL if current["result"] == 1 - me else 0.0
        if self.model is not None:
            probability = self.model.predict(self.featurizer.featurize(current, me))
            return VALUE_SCALE * (2 * probability - 1)
        mine, theirs = current["players"][me], current["players"][1 - me]
        return (
            300.0 * (len(theirs["prize"]) - len(mine["prize"]))
            + self.board(mine)
            - self.board(theirs)
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

    def rival_policy(self, deck: list[int]) -> Policy:
        """Heuristic for the opponent's deck (seen cards plus the sampled hidden ones)."""
        key = tuple(sorted(deck))
        if key == self.own_key:
            return self.policy
        rival = self.rivals.get(key)
        if rival is None:
            if len(self.rivals) >= RIVAL_CACHE:
                self.rivals.clear()
            rival = self.rivals[key] = Policy(deck, self.policy.cards, self.policy.attacks)
        return rival

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
