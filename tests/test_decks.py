import pytest

from assets import validate_deck
from decks import DECKS, deck_list
from engine import Battle, battle_finish, battle_start
from main import ATTACKS, CARDS
from policy import Policy
from schema import Card, Current, Selection


def selection(kind: int, context: int) -> Selection:
    return {
        "type": kind,
        "context": context,
        "minCount": 1,
        "maxCount": 1,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": [],
        "deck": None,
        "contextCard": None,
        "effect": None,
    }


def card(name: str) -> int:
    return next(i for i, data in CARDS.items() if data["name"] == name)


def state(deck: list[int]) -> Current:
    observation, start = battle_start(deck, deck)
    assert start.errorPlayer == -1
    current: Current = observation["current"]
    battle_finish()
    Battle.battle_ptr = None
    return current


@pytest.mark.parametrize("name", sorted(DECKS))
def test_candidate_decks_are_legal_and_accepted_by_the_engine(name: str) -> None:
    deck = deck_list(name)
    validate_deck(deck, CARDS)
    observation, start = battle_start(deck, deck_list("abomasnow"))
    battle_finish()
    Battle.battle_ptr = None
    assert start.errorPlayer == -1
    assert observation["current"] is not None


def test_best_attack_considers_costed_attacks_and_evolution_line() -> None:
    policy = Policy(deck_list("kingambit"), CARDS, ATTACKS)
    current = state(policy.deck)
    pawniard: Card = {"id": card("Pawniard"), "serial": 1, "playerIndex": 0, "energies": [8]}
    attack, missing = policy.best_attack(pawniard, current)
    assert attack is not None and attack["name"] == "Double-Edged Slash"
    assert missing == 1
    assert policy.attach_worth(pawniard, 8, current) > 300


def test_supreme_overlord_scales_with_prizes_taken() -> None:
    policy = Policy(deck_list("kingambit"), CARDS, ATTACKS)
    current = state(policy.deck)
    kingambit: Card = {"id": card("Kingambit"), "serial": 1, "playerIndex": 0, "energies": [8, 8]}
    slash = ATTACKS[CARDS[card("Kingambit")]["attacks"][0]]
    assert policy.estimate(slash, kingambit, current) == 180
    current["players"][1 - current["yourIndex"]]["prize"] = [None] * 4
    assert policy.estimate(slash, kingambit, current) == 240


def test_draw_supporter_is_played_when_the_hand_is_clogged() -> None:
    policy = Policy(deck_list("kingambit"), CARDS, ATTACKS)
    current = state(policy.deck)
    player = current["players"][current["yourIndex"]]
    hand: list[Card] = [{"id": 8, "serial": 10 + i, "playerIndex": 0} for i in range(7)]
    hand.append({"id": card("Cheren"), "serial": 30, "playerIndex": 0})
    player["hand"] = hand
    player["handCount"] = 8
    prompt: Selection = selection(0, 0)
    prompt["option"] = [{"type": 7, "area": 2, "index": 7}, {"type": 14}]
    assert policy.choose({"select": prompt, "current": current}) == [0]
    hand.append({"id": card("Pawniard"), "serial": 31, "playerIndex": 0})
    player["handCount"] = 9
    assert policy.choose({"select": prompt, "current": current}) == [0]
    hand[7] = {"id": card("Lillie's Determination"), "serial": 30, "playerIndex": 0}
    assert policy.choose({"select": prompt, "current": current}) == [1]


def test_field_schedule_splits_games_by_weight_and_interleaves_archetypes() -> None:
    from collections import Counter

    from benchmark import field_schedule

    schedule = field_schedule({"a": 150, "b": 248, "c": 59}, 150)
    assert len(schedule) == 150
    assert Counter(schedule) == {"a": 49, "b": 82, "c": 19}
    assert schedule[:4] == ["b", "a", "b", "c"]
    assert field_schedule({"solo": 5}, 3) == ["solo"] * 3


def test_agent_for_builds_a_full_searcher_per_deck() -> None:
    from benchmark import agent_for
    from main import POLICY, SEARCHER

    assert agent_for(None) == (POLICY, SEARCHER)
    assert agent_for(sorted(POLICY.deck)) == (POLICY, SEARCHER)
    policy, searcher = agent_for(deck_list("kangaskhan"))
    assert policy is not POLICY and sorted(policy.deck) == sorted(deck_list("kangaskhan"))
    assert searcher.policy is policy and searcher.budget == SEARCHER.budget
    assert searcher.model is SEARCHER.model
    assert searcher.tracker is not None and searcher.tracker is not SEARCHER.tracker
    assert agent_for(deck_list("kangaskhan")) == (policy, searcher)


def test_field_pilots_are_separate_from_the_measured_agent() -> None:
    import benchmark
    from main import POLICY

    pilot, searcher = benchmark.pilot_for(list(POLICY.deck))
    assert pilot is not POLICY and searcher.tracker is not None
    assert benchmark.pilot_for(list(POLICY.deck)) == (pilot, searcher)
    assert searcher.model is benchmark.OPPONENT_MODEL
