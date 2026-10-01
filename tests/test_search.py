import copy
from collections import Counter
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
        assert SEARCHER.choose(observation, remaining=60.0) == expected
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


def hand_selection(count: int, low: int, high: int) -> tuple[Selection, Current]:
    hand = [
        cast(Card, {"id": card_id, "serial": index})
        for index, card_id in enumerate([3, 722, 723, 1121][:count])
    ]
    current: Current = {
        "players": [player(hand), player(None)],
        "yourIndex": 0,
        "turn": 3,
        "turnActionCount": 0,
        "result": -1,
        "stadium": [],
        "looking": None,
        "energyAttached": False,
        "supporterPlayed": False,
    }
    selection: Selection = {
        "type": 1,
        "context": 7,
        "minCount": low,
        "maxCount": high,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": [
            {"type": 3, "area": 2, "index": index, "playerIndex": None} for index in range(count)
        ],
        "deck": None,
        "contextCard": None,
        "effect": None,
    }
    return selection, current


def test_candidates_are_distinct_legal_answers_led_by_the_heuristic() -> None:
    searcher = Searcher(POLICY, None, candidates=12)
    selection, current = hand_selection(4, 1, 2)
    fallback = POLICY.choose({"select": selection, "current": current})
    candidates = searcher.candidates_for(selection, current, fallback)
    assert candidates[0] == fallback
    assert len(candidates) > 2
    assert len({tuple(candidate) for candidate in candidates}) == len(candidates)
    for candidate in candidates:
        assert 1 <= len(candidate) <= 2
        assert len(set(candidate)) == len(candidate)
        assert all(0 <= index < 4 for index in candidate)
    optional, current = hand_selection(3, 0, 1)
    fallback = POLICY.choose({"select": optional, "current": current})
    singles = searcher.candidates_for(optional, current, fallback)
    assert [] in singles and all(len(candidate) <= 1 for candidate in singles)
    assert (
        len(Searcher(POLICY, None, candidates=2).candidates_for(optional, current, fallback)) == 2
    )


def test_budget_allocation_follows_the_remaining_overage_pool() -> None:
    searcher = Searcher(POLICY, None, budget=4.0, reserve=60.0, per_game=40, games=3)
    _, current = hand_selection(1, 1, 1)
    assert searcher.allocate(current, None) == 4.0
    assert searcher.allocate(current, 600.0) == pytest.approx(4.0)
    assert searcher.allocate(current, 300.0) == pytest.approx(2.0)
    assert searcher.allocate(current, 60.0) <= 0.0
    assert searcher.allocate({**current, "round": 3}, 300.0) == pytest.approx(4.0)
    assert searcher.allocate({**current, "round": 2}, 300.0) == pytest.approx(3.0)


def test_non_main_prompts_are_searched_only_when_enabled() -> None:
    observation = cast(Observation, battle_start(POLICY.deck, POLICY.deck)[0])
    try:
        while True:
            selection, current = observation["select"], observation["current"]
            assert selection is not None and current is not None
            if current["result"] >= 0:
                pytest.skip("no searchable card prompt reached in this game")
            opponent = current["players"][1 - current["yourIndex"]]
            if (
                selection["type"] == 1
                and current["turn"] >= 2
                and len(selection["option"]) >= 2
                and all(card is not None for card in opponent["active"])
            ):
                break
            observation = cast(Observation, battle_select(POLICY.choose(observation)))
        narrow = Searcher(POLICY, load_engine(), budget=0.1, seed=1, prompts="main")
        assert narrow.choose(observation) == POLICY.choose(observation)
        assert narrow.calls == 0
        broad = Searcher(POLICY, load_engine(), budget=0.1, seed=1, prompts="all", halving=True)
        action = broad.choose(observation)
        assert broad.calls == 1 and broad.samples > 0
        assert selection["minCount"] <= len(action) <= selection["maxCount"]
        assert all(0 <= index < len(selection["option"]) for index in action)
        battle_select(action)
    finally:
        battle_finish()
        Battle.battle_ptr = None


def test_rollouts_model_the_opponent_with_a_policy_for_their_deck() -> None:
    searcher = Searcher(POLICY, None)
    assert searcher.rival_policy(list(POLICY.deck)) is POLICY
    metal = next(i for i, card in POLICY.cards.items() if card["name"] == "Basic {M} Energy")
    zacian = next(i for i, card in POLICY.cards.items() if card["name"] == "Zacian ex")
    rival = searcher.rival_policy([zacian] * 4 + [metal] * 56)
    assert rival is not POLICY
    assert rival.energy_type == POLICY.cards[metal]["energyType"] != POLICY.energy_type
    assert rival.power["Zacian ex"] > 0


def test_ending_the_turn_is_not_searched_while_useful_actions_remain() -> None:
    _, selection, current = main_selection()
    options = selection["option"]
    ending = [i for i, option in enumerate(options) if option["type"] == 14]
    scores = POLICY.scores(selection, current)
    assert ending and max(scores[i] for i in range(len(options)) if i not in ending) > 0
    fallback = POLICY.choose({"select": selection, "current": current})
    gated = Searcher(POLICY, None, candidates=len(options))
    assert ending[0] not in [c[0] for c in gated.candidates_for(selection, current, fallback)]
    free = Searcher(POLICY, None, candidates=len(options), idle="search")
    assert ending[0] in [c[0] for c in free.candidates_for(selection, current, fallback)]


def test_expected_pokemon_follow_the_archetype_posterior() -> None:
    searcher = Searcher(POLICY, None)
    lucario = {i for i, data in POLICY.cards.items() if data["name"] == "Mega Lucario ex"}
    hariyama = {i for i, data in POLICY.cards.items() if data["name"] == "Hariyama"}
    assert not lucario & searcher.expected_pokemon(Counter())
    seen = Counter({card_id: 2 for card_id in lucario} | {card_id: 1 for card_id in hariyama})
    expected = searcher.expected_pokemon(seen)
    assert lucario <= expected and hariyama <= expected
    assert all(POLICY.cards[card_id]["cardType"] == 0 for card_id in expected)
