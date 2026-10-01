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


def test_ex_attackers_do_not_count_damage_against_an_immune_defender() -> None:
    policy = Policy(deck_list("abomasnow"), CARDS, ATTACKS)
    current = state(policy.deck)
    me, them = (
        current["players"][current["yourIndex"]],
        current["players"][1 - current["yourIndex"]],
    )
    mega: Card = {
        "id": card("Mega Abomasnow ex"),
        "serial": 1,
        "playerIndex": 0,
        "energies": [3, 3, 3],
    }
    kyogre: Card = {"id": card("Kyogre"), "serial": 2, "playerIndex": 0, "energies": [3, 3, 3]}
    crustle: Card = {"id": card("Crustle"), "serial": 3, "playerIndex": 1, "hp": 140, "maxHp": 140}
    them["active"] = [crustle]
    me["active"] = [mega]
    assert policy.best_damage(current) == 0
    frost_barrier = CARDS[card("Mega Abomasnow ex")]["attacks"][0]
    me["active"] = [kyogre]
    assert policy.best_damage(current) > 0
    me["active"] = [mega]
    them["active"] = [
        {"id": card("Kyogre"), "serial": 4, "playerIndex": 1, "hp": 140, "maxHp": 140}
    ]
    open_score = policy.attack_score(frost_barrier, current)
    them["active"] = [crustle]
    assert policy.attack_score(frost_barrier, current) < open_score - 100


def test_energy_scaling_attacks_count_attached_energy() -> None:
    policy = Policy(deck_list("hydrapple_ex-teal_mask_ogerpon_ex"), CARDS, ATTACKS)
    current = state(policy.deck)
    me = current["players"][current["yourIndex"]]
    hydrapple: Card = {
        "id": card("Hydrapple ex"),
        "serial": 1,
        "playerIndex": 0,
        "energies": [1, 1],
    }
    ogerpon: Card = {
        "id": card("Teal Mask Ogerpon ex"),
        "serial": 2,
        "playerIndex": 0,
        "energies": [1, 3],
    }
    me["active"] = [hydrapple]
    me["bench"] = [ogerpon]
    syrup_storm = next(
        ATTACKS[i]
        for i in CARDS[card("Hydrapple ex")]["attacks"]
        if ATTACKS[i]["name"] == "Syrup Storm"
    )
    assert policy.estimate(syrup_storm, hydrapple, current) == 30 + 30 * 3
    shower = next(a for a in ATTACKS.values() if a["name"] == "Myriad Leaf Shower")
    current["players"][1 - current["yourIndex"]]["active"] = [
        {"id": card("Kyogre"), "serial": 9, "playerIndex": 1, "energies": [3, 3, 3]}
    ]
    assert policy.estimate(shower, ogerpon, current) == 30 + 30 * 5
    combo = next(a for a in ATTACKS.values() if a["name"] == "Rapid-Fire Combo")
    assert policy.estimate(combo, ogerpon, current) == 250
