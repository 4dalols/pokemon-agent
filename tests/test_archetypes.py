import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from assets import validate_deck
from benchmark import load_main, play_game
from engine import Battle, battle_finish, battle_start
from main import ATTACKS, CARDS
from policy import Policy
from prepare_assets import DECKS, deck_ids
from schema import Card, Current, Selection


@pytest.fixture
def state() -> Current:
    observation, result = battle_start(deck_ids("abomasnow"), deck_ids("abomasnow"))
    assert result.errorPlayer == -1
    current = cast(Current, observation["current"])
    battle_finish()
    Battle.battle_ptr = None
    return current


def prompt(options: list[Card], context: int = 14) -> Selection:
    return {
        "type": 1,
        "context": context,
        "minCount": 1,
        "maxCount": 1,
        "remainDamageCounter": 6,
        "remainEnergyCost": 0,
        "option": [
            {"type": 3, "area": 5, "index": i, "playerIndex": 1} for i in range(len(options))
        ],
        "deck": None,
        "contextCard": None,
        "effect": None,
    }


@pytest.mark.parametrize("name", DECKS)
def test_reconstructed_decks_are_native_legal(name: str) -> None:
    deck = deck_ids(name)
    validate_deck(deck, CARDS)
    try:
        _, result = battle_start(deck, deck)
        assert result.errorPlayer == -1
    finally:
        battle_finish()
        Battle.battle_ptr = None


def test_dragapult_energy_attachment_fills_missing_color() -> None:
    policy = Policy(deck_ids("dragapult"), CARDS, ATTACKS)
    dreepy: Card = {"id": 119, "serial": 0, "playerIndex": 0, "energies": [2]}
    assert policy.energy_deficit(dreepy, 5) == 0
    assert policy.energy_deficit(dreepy, 2) == 1


def test_damage_counters_finish_multiple_knockouts_without_overkill(state: Current) -> None:
    state["yourIndex"] = 0
    bench: list[Card] = [
        {"id": 119, "serial": 1, "playerIndex": 1, "hp": 0},
        {"id": 119, "serial": 2, "playerIndex": 1, "hp": 20},
        {"id": 96, "serial": 3, "playerIndex": 1, "hp": 50},
    ]
    state["players"][1]["bench"] = list(bench)
    policy = Policy(deck_ids("dragapult"), CARDS, ATTACKS)
    selection = prompt(bench)
    assert policy.choose({"select": selection, "current": state}) == [2]
    bench[2]["hp"] = 0
    selection["remainDamageCounter"] = 2
    assert policy.choose({"select": selection, "current": state}) == [1]


@pytest.mark.parametrize(
    "name,card_id", [("hydrapple", 96), ("dragapult", 120), ("kangaskhan", 756)]
)
def test_draw_and_acceleration_abilities_precede_ending_turn(
    state: Current,
    name: str,
    card_id: int,
) -> None:
    player = state["players"][state["yourIndex"]]
    player["active"] = [{"id": card_id, "serial": 1, "playerIndex": state["yourIndex"]}]
    player["hand"] = [{"id": 1, "serial": 2, "playerIndex": state["yourIndex"]}]
    selection = prompt([], 0)
    selection["type"] = 0
    selection["option"] = [{"type": 14}, {"type": 10, "area": 4, "index": 0}]
    policy = Policy(deck_ids(name), CARDS, ATTACKS)
    assert policy.choose({"select": selection, "current": state}) == [1]


def test_hydrapple_damage_counts_grass_across_board(state: Current) -> None:
    policy = Policy(deck_ids("hydrapple"), CARDS, ATTACKS)
    own = state["players"][state["yourIndex"]]
    own["active"] = [{"id": 150, "serial": 1, "playerIndex": 0, "energies": [1, 1]}]
    own["bench"] = []
    before = policy.attack_score(195, state)
    own["bench"] = [{"id": 96, "serial": 2, "playerIndex": 0, "energies": [1, 1, 1]}]
    assert policy.attack_score(195, state) > before


def test_coin_flip_damage_is_not_a_guaranteed_final_knockout(state: Current) -> None:
    policy = Policy(deck_ids("kangaskhan"), CARDS, ATTACKS)
    own = state["players"][state["yourIndex"]]
    opponent = state["players"][1 - state["yourIndex"]]
    own["active"] = [{"id": 756, "serial": 1, "playerIndex": 0}]
    own["prize"] = [None]
    opponent["active"] = [{"id": 96, "serial": 2, "playerIndex": 1, "hp": 250}]
    assert policy.attack_score(1092, state) < 10000
    opponent["active"][0] = {"id": 96, "serial": 2, "playerIndex": 1, "hp": 200}
    assert policy.attack_score(1092, state) > 10000


def test_slowking_values_copied_spread_attack(state: Current) -> None:
    policy = Policy(deck_ids("kangaskhan"), CARDS, ATTACKS)
    opponent = state["players"][1 - state["yourIndex"]]
    opponent["bench"] = [{"id": 119, "serial": i, "playerIndex": 1, "hp": 70} for i in (1, 2)]
    kyurem: Card = {"id": 144, "serial": 3, "playerIndex": 0}
    kangaskhan: Card = {"id": 756, "serial": 4, "playerIndex": 0}
    assert policy.copy_attack_value(kyurem, state) > 300
    assert policy.copy_attack_value(kangaskhan, state) < 0


@pytest.mark.parametrize("name", ["dragapult", "kangaskhan", "hydrapple"])
def test_archetype_completes_native_game(name: str) -> None:
    result = play_game(("greedy", 0, 3), deck=name)
    assert result.winner in (0, 1, 2)


def test_frozen_main_import_is_isolated(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(
        "from policy import marker\ndef agent(obs): return [marker]\n"
    )
    (tmp_path / "policy.py").write_text("marker = 12345\n")
    agent = load_main(tmp_path)
    assert agent({"select": None, "current": None}) == [12345]
    assert Policy(deck_ids("abomasnow"), CARDS, ATTACKS).deck == deck_ids("abomasnow")


def test_deck_environment_selects_exact_playground_list() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from prepare_assets import DECK; "
            "print(sum(DECK.values()), DECK[1120], DECK[1197], DECK[1246])",
        ],
        env={**os.environ, "PTCG_DECK": "dragapult"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().endswith("60 4 2 2")


def test_munkidori_uses_darkness_for_ability_not_psychic_attack() -> None:
    policy = Policy(deck_ids("dragapult"), CARDS, ATTACKS)
    card: Card = {"id": 112, "serial": 0, "playerIndex": 0}
    assert policy.energy_deficit(card, 7) == 0
    assert policy.energy_deficit(card, 5) == 1


def test_damage_counters_choose_two_knockouts_over_one_equal_prize_target(state: Current) -> None:
    state["yourIndex"] = 0
    bench: list[Card] = [
        {"id": 119, "serial": 1, "playerIndex": 1, "hp": 20},
        {"id": 119, "serial": 2, "playerIndex": 1, "hp": 20},
        {"id": 96, "serial": 3, "playerIndex": 1, "hp": 60},
    ]
    state["players"][1]["bench"] = list(bench)
    policy = Policy(deck_ids("dragapult"), CARDS, ATTACKS)
    assert policy.choose({"select": prompt(bench), "current": state}) == [0]


def test_munkidori_ability_rejects_empty_damage_transfer(state: Current) -> None:
    policy = Policy(deck_ids("dragapult"), CARDS, ATTACKS)
    card: Card = {"id": 112, "serial": 0, "playerIndex": 0, "hp": 110, "maxHp": 110}
    own = state["players"][state["yourIndex"]]
    own["active"], own["bench"] = [card], []
    selection = prompt([], 0)
    selection["option"] = [{"type": 10, "area": 4, "index": 0}, {"type": 14}]
    assert policy.choose({"select": selection, "current": state}) == [1]
    card["hp"] = 80
    assert policy.choose({"select": selection, "current": state}) == [0]


def test_boss_values_two_prize_knockout_over_one(state: Current) -> None:
    policy = Policy(deck_ids("dragapult"), CARDS, ATTACKS)
    own = state["players"][state["yourIndex"]]
    own["active"] = [{"id": 121, "serial": 0, "playerIndex": 0, "energies": [2, 5]}]
    basic: Card = {"id": 119, "serial": 1, "playerIndex": 1, "hp": 30}
    ex: Card = {"id": 96, "serial": 2, "playerIndex": 1, "hp": 190}
    assert policy.gust_value(ex, state) > policy.gust_value(basic, state)
