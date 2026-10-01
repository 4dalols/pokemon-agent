import pytest

from assets import validate_deck
from decks import DECKS, deck_list
from engine import Battle, battle_finish, battle_start
from main import ATTACKS, CARDS
from memory import Threat
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
    policy, searcher = agent_for(deck_list("lucario"))
    assert policy is not POLICY and sorted(policy.deck) == sorted(deck_list("lucario"))
    assert searcher.policy is policy and searcher.budget == SEARCHER.budget
    assert searcher.model is SEARCHER.model
    assert searcher.tracker is not None and searcher.tracker is not SEARCHER.tracker
    assert agent_for(deck_list("lucario")) == (policy, searcher)


def test_field_pilots_are_separate_from_the_measured_agent() -> None:
    import benchmark
    from main import POLICY

    pilot, searcher = benchmark.pilot_for(list(POLICY.deck))
    assert pilot is not POLICY and searcher.tracker is not None
    assert benchmark.pilot_for(list(POLICY.deck)) == (pilot, searcher)
    assert searcher.model is benchmark.OPPONENT_MODEL


def test_putting_a_hand_card_back_on_the_deck_is_worse_than_ending_the_turn() -> None:
    policy = Policy(deck_list("kangaskhan"), CARDS, ATTACKS)
    current = state(policy.deck)
    current["stadium"] = [{"id": card("Academy at Night"), "serial": 99, "playerIndex": 1}]
    main = selection(0, 0)
    main["option"] = [{"type": 10, "area": 7, "index": 0}, {"type": 14}]
    scores = policy.scores(main, current)
    assert scores[0] < scores[1]


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


def test_ability_pokemon_cannot_damage_an_ability_walled_defender() -> None:
    policy = Policy(deck_list("kangaskhan"), CARDS, ATTACKS)
    current = state(policy.deck)
    me, them = (
        current["players"][current["yourIndex"]],
        current["players"][1 - current["yourIndex"]],
    )
    ogerpon: Card = {
        "id": card("Cornerstone Mask Ogerpon ex"),
        "serial": 1,
        "playerIndex": 1,
        "hp": 210,
        "maxHp": 210,
    }
    kangaskhan: Card = {
        "id": card("Mega Kangaskhan ex"),
        "serial": 2,
        "playerIndex": 0,
        "energies": [11, 11, 11],
    }
    dwebble: Card = {"id": card("Dwebble"), "serial": 3, "playerIndex": 0, "energies": [11]}
    them["active"] = [ogerpon]
    me["active"] = [kangaskhan]
    assert policy.best_damage(current) == 0
    me["active"] = [dwebble]
    assert policy.best_damage(current) == 0  # Ascension deals no damage
    assert not policy.immune(ogerpon, dwebble)
    assert policy.immune(ogerpon, kangaskhan)


def lucario_board(policy: Policy) -> tuple[Current, Card, Card]:
    current = state(policy.deck)
    me, them = (
        current["players"][current["yourIndex"]],
        current["players"][1 - current["yourIndex"]],
    )
    kangaskhan: Card = {
        "id": card("Mega Kangaskhan ex"),
        "serial": 1,
        "playerIndex": 0,
        "hp": 160,
        "maxHp": 300,
        "energies": [11],
    }
    crustle: Card = {
        "id": card("Crustle"),
        "serial": 2,
        "playerIndex": 0,
        "hp": 150,
        "maxHp": 150,
        "energies": [],
    }
    me["active"] = [kangaskhan]
    me["bench"] = [crustle]
    them["active"] = [{"id": card("Riolu"), "serial": 3, "playerIndex": 1, "hp": 80, "maxHp": 80}]
    them["bench"] = []
    return current, kangaskhan, crustle


def test_known_attackers_make_a_doomed_active_a_poor_energy_target() -> None:
    policy = Policy(deck_list("kangaskhan"), CARDS, ATTACKS)
    current, kangaskhan, crustle = lucario_board(policy)
    main = selection(0, 0)
    energy = next(i for i, c in enumerate(policy.deck) if CARDS[c]["name"] == "Mist Energy")
    current["players"][current["yourIndex"]]["hand"] = [
        {"id": energy, "serial": 9, "playerIndex": 0}
    ]
    main["option"] = [
        {"type": 8, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},
        {"type": 8, "area": 2, "index": 0, "inPlayArea": 5, "inPlayIndex": 0},
    ]
    assert not policy.exposed(kangaskhan, current)
    relaxed = policy.scores(main, current)
    assert relaxed[0] > relaxed[1]
    policy.threat = Threat(attackers=(card("Mega Lucario ex"),))
    assert policy.exposed(kangaskhan, current)
    assert not policy.exposed(crustle, current)
    pressed = policy.scores(main, current)
    assert pressed[1] > pressed[0] > 0
    assert policy.readiness(kangaskhan, current) < policy.readiness(crustle, current)
    policy.threat = Threat(attackers=(card("Hariyama"),))
    assert policy.exposed(crustle, current)
    policy.threat = Threat(attackers=(card("Riolu"),))
    assert not policy.exposed(crustle, current) and not policy.exposed(kangaskhan, current)


def test_multi_prize_basics_stay_in_hand_against_a_known_one_hit_knockout() -> None:
    policy = Policy(deck_list("kangaskhan"), CARDS, ATTACKS)
    current, _, _ = lucario_board(policy)
    ogerpon = card("Cornerstone Mask Ogerpon ex")
    current["players"][current["yourIndex"]]["hand"] = [
        {"id": ogerpon, "serial": 9, "playerIndex": 0}
    ]
    main = selection(0, 0)
    main["option"] = [{"type": 7, "area": 2, "index": 0}]
    assert policy.scores(main, current)[0] > 0
    policy.threat = Threat(attackers=(card("Mega Lucario ex"),))
    assert policy.scores(main, current)[0] == -120
    current["players"][current["yourIndex"]]["bench"] = []
    assert policy.scores(main, current)[0] > 0
