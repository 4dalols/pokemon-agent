import copy
import time
from typing import cast

import pytest
from test_search import main_selection

from engine import Battle, battle_finish
from lethal import END_TURN, LethalSolver, state_key
from main import POLICY
from schema import Card, Current, Observation, Player, SearchState, Selection
from search import Searcher, load_engine


def player(deck: int = 30, bench: int = 1, prizes: int = 1) -> Player:
    active = cast(Card, {"id": 721, "serial": 1, "playerIndex": 0, "hp": 150, "maxHp": 150})
    benched = cast(Card, {"id": 722, "serial": 2, "playerIndex": 0, "hp": 70, "maxHp": 70})
    return {
        "deckCount": deck,
        "handCount": 2,
        "hand": [cast(Card, {"id": 3, "serial": 3, "playerIndex": 0})],
        "discard": [],
        "active": [active],
        "bench": [benched] * bench,
        "benchMax": 5,
        "prize": [None] * prizes,
    }


def current(me: int = 0, turn: int = 5, result: int = -1, enemy_deck: int = 30) -> Current:
    return {
        "players": [player(), player(deck=enemy_deck)],
        "yourIndex": me,
        "turn": turn,
        "turnActionCount": 0,
        "result": result,
        "stadium": [],
        "looking": None,
        "energyAttached": False,
        "supporterPlayed": False,
    }


def selection(kind: int, context: int, types: list[int]) -> Selection:
    return {
        "type": kind,
        "context": context,
        "minCount": 1,
        "maxCount": 1,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": [{"type": option} for option in types],
        "deck": None,
        "contextCard": None,
        "effect": None,
    }


MAIN = selection(0, 0, [0, 0, END_TURN])
COIN = selection(9, 3, [1, 2])
PICK = selection(1, 0, [0, 0])
PICK["option"][1]["index"] = 1


class FakeTree:
    """Scripted search tree: node 0 is the live main selection of player 0 on turn 5.

    option 0 -> coin flip: heads wins, tails ends the turn
    option 1 -> the opponent wins (or, with `deep`, a follow-up pick: [0] loses, [1] wins)
    option 2 (End) -> wins only when the opponent is decked out, else the turn ends
    """

    def __init__(self, enemy_deck: int = 30, deep: bool = False) -> None:
        end = current(result=0) if enemy_deck == 0 else current(turn=6)
        self.nodes: dict[int, Observation] = {
            0: {"select": MAIN, "current": current(enemy_deck=enemy_deck)},
            1: {"select": COIN, "current": current(enemy_deck=enemy_deck)},
            2: {"select": None, "current": current(result=0)},
            3: {"select": None, "current": current(turn=6)},
            4: {"select": None, "current": current(result=1)},
            5: {"select": None, "current": end},
            6: {"select": PICK, "current": current()},
        }
        self.edges: dict[tuple[int, tuple[int, ...]], int] = {
            (0, (0,)): 1,
            (1, (0,)): 2,
            (1, (1,)): 3,
            (0, (1,)): 6 if deep else 4,
            (6, (0,)): 4,
            (6, (1,)): 2,
            (0, (2,)): 5,
        }
        self.steps = 0
        self.live: set[int] = set()

    def root(self) -> SearchState:
        return {"observation": self.nodes[0], "searchId": 0}

    def step(self, search_id: int, select: list[int]) -> SearchState | None:
        self.steps += 1
        target = self.edges.get((search_id, tuple(select)))
        if target is None:
            return None
        self.live.add(target)
        return {"observation": self.nodes[target], "searchId": target}

    def release(self, search_id: int) -> None:
        self.live.remove(search_id)


def test_coin_flips_are_averaged_and_threshold_applies() -> None:
    tree = FakeTree()
    solver = LethalSolver(POLICY, tree, threshold=0.5)
    assert solver.solve([tree.root()], [0, 1, 2], deadline=float("inf")) == (0, 0.5)
    assert solver.stats.fired == 1 and solver.stats.calls == 1
    assert solver.stats.probabilities == [0.5]
    assert solver.plan == [], "a line that starts with a coin flip has no replayable prompts"
    assert tree.live == set(), "every explored node must be released"
    strict = LethalSolver(POLICY, FakeTree())
    assert strict.threshold == 1.0
    assert strict.solve([tree.root()], [0, 1, 2], deadline=float("inf")) is None


def test_end_turn_is_tried_when_opponent_is_decked_out() -> None:
    tree = FakeTree(enemy_deck=0)
    solver = LethalSolver(POLICY, tree)
    assert solver.solve([tree.root()], [0, 1, 2], deadline=float("inf")) == (2, 1.0)
    without = LethalSolver(POLICY, FakeTree(), threshold=0.5)
    assert without.solve([FakeTree().root()], [0, 2], deadline=float("inf")) == (0, 0.5)


def test_solved_line_is_recorded_and_replayed_for_follow_up_prompts() -> None:
    tree = FakeTree(deep=True)
    solver = LethalSolver(POLICY, tree)
    assert solver.solve([tree.root()], [0, 1, 2], deadline=float("inf")) == (1, 1.0)
    assert solver.plan == [(1, 2, [1])]
    assert tree.live == set()
    searcher = Searcher(POLICY, None)
    searcher.plan, searcher.plan_turn = list(solver.plan), 5
    assert searcher.choose({"select": PICK, "current": current()}) == [1]
    assert searcher.plan == []
    searcher.plan, searcher.plan_turn = list(solver.plan), 5
    assert searcher.choose({"select": PICK, "current": current(turn=6)}) == POLICY.choose(
        {"select": PICK, "current": current(turn=6)}
    ), "a stale plan from an earlier turn is dropped"
    assert searcher.plan == []
    searcher.plan, searcher.plan_turn = [(1, 3, [1])], 5
    assert searcher.choose({"select": PICK, "current": current()}) == POLICY.choose(
        {"select": PICK, "current": current()}
    ), "a prompt that no longer matches the solved line is dropped"
    assert searcher.plan == []


def test_threat_reports_opponent_win_probability_and_node_cap() -> None:
    tree = FakeTree()
    solver = LethalSolver(POLICY, tree)
    assert solver.threat(tree.root(), deadline=float("inf")) == 0.5
    assert solver.stats.threats == 1 and solver.stats.threats_found == 1
    capped = LethalSolver(POLICY, FakeTree(), threat_node_cap=1)
    assert capped.threat(FakeTree().root(), deadline=float("inf")) == 0.0
    assert solver.solve([tree.root()], [0], deadline=0.0) is None


def test_can_finish_gates_on_prizes_bench_and_deck_out() -> None:
    solver = LethalSolver(POLICY, FakeTree())
    state = current()
    assert solver.can_finish(state, 0), "one prize left and a single-prize defender"
    state["players"][0]["prize"] = [None] * 3
    assert not solver.can_finish(state, 0)
    state["players"][1]["bench"] = []
    assert solver.can_finish(state, 0), "no bench: any knockout wins"
    state["players"][1]["bench"] = [None]
    state["players"][1]["deckCount"] = 0
    assert solver.can_finish(state, 0), "deck-out"
    state["players"][1]["deckCount"] = 5
    state["players"][1]["active"] = []
    assert not solver.can_finish(state, 0)


def test_state_key_ignores_option_order_but_not_board() -> None:
    state = current()
    key = state_key(state, MAIN, 0)
    shuffled = copy.deepcopy(state)
    shuffled["players"][0]["hand"] = list(reversed(shuffled["players"][0]["hand"] or []))
    assert state_key(shuffled, MAIN, 0) == key
    damaged = copy.deepcopy(state)
    damaged["players"][1]["active"][0]["hp"] = 10  # type: ignore[index]
    assert state_key(damaged, MAIN, 0) != key


def test_solver_runs_before_rollouts_on_the_native_engine() -> None:
    engine = load_engine()
    if engine is None:
        pytest.skip("native engine unavailable")
    try:
        observation, selection, state = main_selection()
        searcher = Searcher(POLICY, engine, budget=0.4, seed=3)
        action = searcher.choose(observation)
        assert len(action) == 1 and 0 <= action[0] < len(selection["option"])
        gated = searcher.solver.can_finish(state, state["yourIndex"])
        assert searcher.solver.stats.calls == int(gated)
        assert searcher.samples > 0, "rollouts still run after an unsuccessful solve"
        serialized = observation.get("search_begin_input")
        assert serialized
        scores = POLICY.scores(selection, state)
        hit = searcher.lethal(state, serialized, scores, time.perf_counter() + 0.3)
        assert hit is None or 0 <= hit < len(selection["option"])
        assert searcher.solver.stats.calls == int(gated) + 1
        assert searcher.solver.stats.nodes > 0
        disabled = Searcher(POLICY, engine, budget=0.2, seed=3, lethal_budget=0.0)
        disabled.choose(observation)
        assert disabled.solver.stats.calls == 0
    finally:
        battle_finish()
        Battle.battle_ptr = None
