import copy
from typing import cast

import pytest

from engine import Battle, battle_finish, battle_select, battle_start
from main import POLICY, SEARCHER
from policy import Policy
from schema import Card, Current, Observation, Player, Selection
from search import TERMINAL, Searcher, load_engine


def main_selection() -> tuple[Observation, Selection, Current]:
    observation = cast(Observation, battle_start(POLICY.deck, POLICY.deck)[0])
    while True:
        selection, current = observation["select"], observation["current"]
        assert selection is not None and current is not None
        if current["result"] >= 0:
            pytest.fail("no main selection reached")
        if selection["type"] == 0 and current["turn"] >= 2 and len(selection["option"]) > 1:
            return copy.deepcopy((observation, selection, current))
        observation = cast(Observation, battle_select(POLICY.choose(observation)))


def player(hand: list[Card] | None) -> Player:
    return {
        "deckCount": 40,
        "handCount": 1,
        "hand": hand,
        "discard": [],
        "active": [],
        "bench": [],
        "benchMax": 5,
        "prize": [None] * 6,
    }


def test_engine_is_available_and_declares_search_abi() -> None:
    engine = load_engine()
    assert engine is not None
    assert engine.SearchStep.restype is not None


def test_search_returns_a_single_legal_main_option() -> None:
    try:
        observation, selection, _ = main_selection()
        searcher = Searcher(POLICY, load_engine(), budget=0.1, seed=1)
        action = searcher.choose(observation)
        assert len(action) == 1
        assert 0 <= action[0] < len(selection["option"])
        assert searcher.samples > 0
        battle_select(action)
    finally:
        battle_finish()
        Battle.battle_ptr = None


def test_search_falls_back_to_heuristics_without_engine_or_budget() -> None:
    try:
        observation, _, _ = main_selection()
        expected = POLICY.choose(observation)
        assert Searcher(POLICY, None).choose(observation) == expected
        assert SEARCHER.choose(observation, remaining=100.0) == expected
        assert SEARCHER.choose({"select": None, "current": None}) == POLICY.deck
    finally:
        battle_finish()
        Battle.battle_ptr = None


def test_explicit_null_player_index_resolves_to_own_cards() -> None:
    current: Current = {
        "players": [player([cast(Card, {"id": 721, "hp": 150, "maxHp": 150})]), player(None)],
        "yourIndex": 0,
        "turn": 1,
        "turnActionCount": 0,
        "result": -1,
        "stadium": [],
        "looking": None,
        "energyAttached": False,
        "supporterPlayed": False,
    }
    selection: Selection = {
        "type": 1,
        "context": 22,
        "minCount": 1,
        "maxCount": 1,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": [{"type": 3, "area": 2, "index": 0, "playerIndex": None}],
        "deck": None,
        "contextCard": None,
        "effect": None,
    }
    card = POLICY.resolve(selection["option"][0], selection, current)
    assert card is not None and card["id"] == 721
    assert Policy.player_index({"type": 3, "playerIndex": 1}, current) == 1


def test_terminal_outcomes_dominate_board_evaluation() -> None:
    searcher = Searcher(POLICY, None)
    observation, _, current = main_selection()
    battle_finish()
    Battle.battle_ptr = None
    me = current["yourIndex"]
    won = copy.deepcopy(observation)
    won["current"] = {**current, "result": me}
    lost = copy.deepcopy(observation)
    lost["current"] = {**current, "result": 1 - me}
    assert searcher.evaluate(won, me) == TERMINAL
    assert searcher.evaluate(lost, me) == -TERMINAL
    assert abs(searcher.evaluate(observation, me)) < TERMINAL


def test_rollouts_model_the_opponent_with_a_policy_for_their_deck() -> None:
    searcher = Searcher(POLICY, None)
    assert searcher.rival_policy(list(POLICY.deck)) is POLICY
    metal = next(i for i, card in POLICY.cards.items() if card["name"] == "Basic {M} Energy")
    zacian = next(i for i, card in POLICY.cards.items() if card["name"] == "Zacian ex")
    rival = searcher.rival_policy([zacian] * 4 + [metal] * 56)
    assert rival is not POLICY
    assert rival.energy_type == POLICY.cards[metal]["energyType"] != POLICY.energy_type
    assert rival.power["Zacian ex"] > 0
