"""Exact within-turn lethal search on the native engine.

Our hand and board are public to us, so the only uncertainty inside the current turn
is the order of our own deck (draws, deck searches) and coin flips. `LethalSolver.solve`
enumerates the current turn's action tree depth-first for several determinizations of
the hidden cards, pruning transpositions on the resulting public state. Coins are
enumerated exactly: the roots are opened with `manualCoin`, so every flip arrives as a
Yes/No prompt whose two branches are averaged. A line is scored by P(win this turn) and
the first action of the best line is returned when it clears the threshold.

`LethalSolver.win_probability` is the same search run for whichever player is about to
act; the rollout search uses it as a one-ply "can the opponent finish us in their
reply" check.
"""

import time
from dataclasses import dataclass, field
from typing import Protocol

from policy import Policy
from schema import Card, Current, Observation, Player, SearchState, Selection

MAIN = 0
EVOLVE = 8
YES_NO = 9
COIN_HEAD = 3
TO_HAND = 7
END_TURN = 14
StateKey = tuple[object, ...]


class SearchTree(Protocol):
    def step(self, search_id: int, select: list[int]) -> SearchState | None: ...

    def release(self, search_id: int) -> None: ...


@dataclass
class SolverStats:
    calls: int = 0
    fired: int = 0
    threats: int = 0
    threats_found: int = 0
    nodes: int = 0
    seconds: float = 0.0
    probabilities: list[float] = field(default_factory=list)


class Budget:
    def __init__(self, deadline: float, node_cap: int) -> None:
        self.deadline = deadline
        self.nodes = 0
        self.node_cap = node_cap

    def spent(self) -> bool:
        return self.nodes >= self.node_cap or time.perf_counter() >= self.deadline


class LethalSolver:
    def __init__(
        self,
        policy: Policy,
        tree: SearchTree,
        determinizations: int = 4,
        node_cap: int = 3000,
        threat_node_cap: int = 300,
        alternatives: int = 4,
        threshold: float = 0.5,
    ) -> None:
        self.policy = policy
        self.tree = tree
        self.determinizations = determinizations
        self.node_cap = node_cap
        self.threat_node_cap = threat_node_cap
        self.alternatives = alternatives
        self.threshold = threshold
        self.stats = SolverStats()

    def prize_value(self, card: Card | None) -> int:
        if card is None or card["id"] not in self.policy.cards:
            return 1
        data = self.policy.cards[card["id"]]
        return 3 if data["megaEx"] else 2 if data["ex"] else 1

    def can_finish(self, current: Current, player: int) -> bool:
        """Can `player` plausibly end the game this turn (last prizes, lone active, deck-out)?"""
        mine, theirs = current["players"][player], current["players"][1 - player]
        if theirs["deckCount"] == 0:
            return True
        defending = theirs["active"][0] if theirs["active"] else None
        if defending is None:
            return False
        if not any(card is not None for card in theirs["bench"]):
            return True
        return len(mine["prize"]) <= self.prize_value(defending)

    def solve(
        self, roots: list[SearchState], first: list[int], deadline: float
    ) -> tuple[int, float] | None:
        """Return (option index, P(win this turn)) for the best opening action.

        `roots` holds one search node per determinization of the same live selection;
        `first` lists the root option indices worth trying, best heuristic first.
        """
        started = time.perf_counter()
        self.stats.calls += 1
        wins = {index: 0.0 for index in first}
        completed = 0
        for root in roots:
            current = root["observation"]["current"]
            if current is None or time.perf_counter() >= deadline:
                break
            budget = Budget(deadline, self.node_cap)
            me = current["yourIndex"]
            seen: dict[StateKey, float] = {}
            for index in first:
                budget.nodes += 1
                child = self.tree.step(root["searchId"], [index])
                if child is None:
                    continue
                wins[index] += self.win_probability(child, me, current["turn"], budget, seen)
                self.tree.release(child["searchId"])
                if budget.spent():
                    break
            self.stats.nodes += budget.nodes
            if budget.spent() and max(wins.values()) < completed + 1:
                break
            completed += 1
        self.stats.seconds += time.perf_counter() - started
        if completed == 0:
            return None
        best = max(first, key=lambda index: (wins[index], -first.index(index)))
        probability = wins[best] / completed
        self.stats.probabilities.append(probability)
        if probability < self.threshold or wins[best] <= 0:
            return None
        self.stats.fired += 1
        return best, probability

    def threat(self, node: SearchState, deadline: float) -> float:
        """P(the player about to act wins before this turn ends), under a small node cap."""
        current = node["observation"]["current"]
        if current is None:
            return 0.0
        self.stats.threats += 1
        budget = Budget(deadline, self.threat_node_cap)
        found = self.win_probability(node, current["yourIndex"], current["turn"], budget, {})
        self.stats.nodes += budget.nodes
        self.stats.threats_found += int(found > 0)
        return found

    def win_probability(
        self,
        node: SearchState,
        me: int,
        turn: int,
        budget: Budget,
        seen: dict[StateKey, float],
    ) -> float:
        observation, search_id = node["observation"], node["searchId"]
        current, selection = observation["current"], observation["select"]
        if current is None:
            return 0.0
        if current["result"] >= 0:
            return 1.0 if current["result"] == me else 0.0
        if selection is None or current["turn"] != turn:
            return 0.0
        if selection["type"] == YES_NO and selection["context"] == COIN_HEAD:
            total = 0.0
            for index in range(len(selection["option"])):
                if budget.spent():
                    break
                budget.nodes += 1
                child = self.tree.step(search_id, [index])
                if child is None:
                    continue
                total += self.win_probability(child, me, turn, budget, seen)
                self.tree.release(child["searchId"])
            return total / len(selection["option"])
        if current["yourIndex"] != me:
            budget.nodes += 1
            child = self.tree.step(search_id, self.policy.choose(observation))
            if child is None:
                return 0.0
            found = self.win_probability(child, me, turn, budget, seen)
            self.tree.release(child["searchId"])
            return found
        key = state_key(current, selection, me)
        cached = seen.get(key)
        if cached is not None:
            return cached
        best = 0.0
        for choice in self.choices(observation, selection, current):
            if budget.spent():
                break
            budget.nodes += 1
            child = self.tree.step(search_id, choice)
            if child is None:
                continue
            best = max(best, self.win_probability(child, me, turn, budget, seen))
            self.tree.release(child["searchId"])
            if best >= 1.0:
                break
        seen[key] = best
        return best

    def choices(
        self, observation: Observation, selection: Selection, current: Current
    ) -> list[list[int]]:
        options = selection["option"]
        scores = self.policy.scores(selection, current)
        ranked = self.policy.rank(scores)
        if selection["type"] == MAIN:
            picks = [[index] for index in ranked if options[index]["type"] != END_TURN]
            if self.opponent(current)["deckCount"] == 0:
                picks.extend([index] for index in ranked if options[index]["type"] == END_TURN)
            return picks
        if selection["context"] == TO_HAND and selection["minCount"] == len(options):
            return [list(range(len(options)))]
        if selection["type"] == YES_NO:
            return [[index] for index in ranked]
        if selection["type"] == EVOLVE:
            return [[index] for index in ranked[: self.alternatives]]
        if selection["maxCount"] == 1 and selection["minCount"] <= 1:
            seen_cards: set[tuple[int, int]] = set()
            picks = []
            for index in ranked:
                card = self.policy.resolve(options[index], selection, current)
                mark = (card["id"], options[index].get("area", 0)) if card else (-index, 0)
                if mark in seen_cards:
                    continue
                seen_cards.add(mark)
                picks.append([index])
                if len(picks) >= self.alternatives:
                    break
            if selection["minCount"] == 0:
                picks.append([])
            return picks
        default = self.policy.choose(observation)
        reverse = list(reversed(ranked))[: len(default)]
        picks = [default]
        if sorted(reverse) != sorted(default) and len(reverse) >= selection["minCount"]:
            picks.append(reverse)
        return picks

    @staticmethod
    def opponent(current: Current) -> Player:
        return current["players"][1 - current["yourIndex"]]


def card_key(card: Card | None) -> tuple[object, ...] | None:
    if card is None:
        return None
    return (
        card["id"],
        card.get("hp"),
        tuple(sorted(card.get("energies", []))),
        tuple(tool["id"] for tool in card.get("tools", [])),
    )


def player_key(player: Player) -> tuple[object, ...]:
    return (
        tuple(sorted(card["id"] for card in player["hand"] or [])),
        tuple(card_key(card) for card in player["active"]),
        tuple(sorted(str(card_key(card)) for card in player["bench"] if card is not None)),
        player["deckCount"],
        player["handCount"],
        len(player["discard"]),
        len(player["prize"]),
    )


def state_key(current: Current, selection: Selection, me: int) -> StateKey:
    return (
        player_key(current["players"][me]),
        player_key(current["players"][1 - me]),
        current["energyAttached"],
        current["supporterPlayed"],
        tuple(card["id"] for card in current["stadium"]),
        selection["type"],
        selection["context"],
        len(selection["option"]),
        selection["minCount"],
        selection["maxCount"],
    )
