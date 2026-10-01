import copy

import pytest
from kaggle_environments import make

from assets import ROOT, validate_deck
from benchmark import play_game
from engine import Battle, battle_finish, battle_start
from main import ATTACKS, CARDS, POLICY, agent
from schema import Current, Observation, Selection


@pytest.fixture
def current() -> Current:
    observation, start = battle_start(POLICY.deck, POLICY.deck)
    assert start.errorPlayer == -1
    state: Current = copy.deepcopy(observation["current"])
    battle_finish()
    Battle.battle_ptr = None
    return state


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


def test_energy_payment_is_an_iterative_single_selection(current: Current) -> None:
    prompt = selection(4, 30)
    prompt["remainEnergyCost"] = 2
    prompt["option"] = [{"type": 6, "count": 1}, {"type": 6, "count": 1}]
    action = agent({"select": prompt, "current": current})
    assert len(action) == 1


def test_count_returns_option_position_not_quantity(current: Current) -> None:
    prompt = selection(8, 38)
    prompt["option"] = [{"type": 0, "number": 2}, {"type": 0, "number": 5}]
    assert agent({"select": prompt, "current": current}) == [1]


def test_facedown_prizes_do_not_need_card_metadata(current: Current) -> None:
    player = current["players"][current["yourIndex"]]
    player["prize"] = [None] * 6
    prompt = selection(1, 7)
    prompt["minCount"] = prompt["maxCount"] = 3
    prompt["option"] = [{"type": 3, "area": 6, "index": index} for index in range(6)]
    action = agent({"select": prompt, "current": current})
    assert len(action) == len(set(action)) == 3
    assert all(0 <= index < 6 for index in action)


def test_optional_search_can_decline_unknown_cards(current: Current) -> None:
    prompt = selection(1, 7)
    prompt["minCount"] = 0
    prompt["option"] = [{"type": 3, "area": 1, "index": 0}]
    assert agent({"select": prompt, "current": current}) == []


def test_missing_optional_fields_and_opponent_hand_are_safe(current: Current) -> None:
    prompt = selection(9, 41)
    prompt["option"] = [{"type": 1}, {"type": 2}]
    current["players"][1 - current["yourIndex"]]["hand"] = None
    assert agent({"select": prompt, "current": current}) == [0]


def test_powerglass_accepts_optional_discard_energy_attachment(current: Current) -> None:
    current["players"][current["yourIndex"]]["discard"] = [
        {"id": 3, "serial": 5, "playerIndex": current["yourIndex"]}
    ]
    prompt = selection(1, 22)
    prompt["minCount"] = 0
    prompt["option"] = [{"type": 3, "area": 3, "index": 0}]
    assert agent({"select": prompt, "current": current}) == [0]


def test_initialization_returns_an_independent_valid_deck() -> None:
    initial: Observation = {"select": None, "current": None}
    submitted = agent(initial)
    validate_deck(submitted, CARDS)
    submitted.clear()
    assert len(agent(initial)) == 60


def test_prepare_a_backup_before_a_nonwinning_attack(current: Current) -> None:
    mega = next(i for i, card in CARDS.items() if card["name"] == "Mega Abomasnow ex")
    kyogre = next(i for i, card in CARDS.items() if card["name"] == "Kyogre")
    attack = next(i for i, data in ATTACKS.items() if data["name"] == "Hammer-lanche")
    player = current["players"][current["yourIndex"]]
    player["active"] = [{"id": mega, "serial": 1, "playerIndex": 0, "energies": [3, 3]}]
    player["bench"] = [{"id": kyogre, "serial": 2, "playerIndex": 0, "energies": []}]
    player["discard"] = [{"id": 3, "serial": i + 20, "playerIndex": 0} for i in range(8)]
    player["prize"] = [None] * 6
    player["deckCount"] = 30
    player["hand"] = [{"id": 3, "serial": 10, "playerIndex": 0}]
    prompt = selection(0, 0)
    prompt["option"] = [
        {"type": 13, "attackId": attack},
        {"type": 8, "area": 2, "index": 0, "inPlayArea": 5, "inPlayIndex": 0},
    ]
    assert agent({"select": prompt, "current": current}) == [1]


def test_guaranteed_final_prize_knockout_beats_optional_preparation(current: Current) -> None:
    mega = next(i for i, card in CARDS.items() if card["name"] == "Mega Abomasnow ex")
    snover = next(i for i, card in CARDS.items() if card["name"] == "Snover")
    frost = next(i for i, attack in ATTACKS.items() if attack["name"] == "Frost Barrier")
    own = current["players"][current["yourIndex"]]
    opponent = current["players"][1 - current["yourIndex"]]
    own["prize"] = [None]
    own["active"] = [{"id": mega, "serial": 1, "playerIndex": 0, "energies": [3, 3, 3]}]
    own["bench"] = [{"id": snover, "serial": 2, "playerIndex": 0, "energies": []}]
    own["hand"] = [{"id": 3, "serial": 10, "playerIndex": 0}]
    opponent["active"] = [{"id": snover, "serial": 3, "playerIndex": 1, "hp": 90}]
    prompt = selection(0, 0)
    prompt["option"] = [
        {"type": 8, "area": 2, "index": 0, "inPlayArea": 5, "inPlayIndex": 0},
        {"type": 13, "attackId": frost},
    ]
    assert agent({"select": prompt, "current": current}) == [1]


def test_deck_validation_rejects_unknown_ids_and_copy_limits() -> None:
    with pytest.raises(ValueError, match="Unknown card ID"):
        validate_deck([999999] * 60, CARDS)
    snover = next(card_id for card_id, data in CARDS.items() if data["name"] == "Snover")
    with pytest.raises(ValueError, match="four copies"):
        validate_deck([snover] * 5 + [3] * 55, CARDS)


@pytest.mark.parametrize("opponent,seat", [("first", 0), ("random", 1), ("greedy", 0), ("self", 1)])
def test_native_games_finish_without_illegal_actions(opponent: str, seat: int) -> None:
    result = play_game((opponent, seat, 42))
    assert result.winner in (0, 1, 2)
    assert result.steps < 10000
    assert len(agent({"select": None, "current": None})) == 60


def test_kaggle_runner_loads_file_without_file_global_and_restarts() -> None:
    for _ in range(2):
        environment = make("cabt", debug=True)
        environment.run([str(ROOT / "main.py"), "random"])
        assert all(state.status == "DONE" for state in environment.state)


def test_kaggle_best_of_three_match_completes() -> None:
    environment = make("cabt", configuration={"bo": 3}, debug=True)
    environment.run([str(ROOT / "main.py"), str(ROOT / "main.py")])
    assert all(state.status == "DONE" for state in environment.state)
    assert sum(environment.result) >= 2
