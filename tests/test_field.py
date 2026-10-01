import os
from pathlib import Path

import pytest

from assets import ROOT, load_deck, validate_deck
from benchmark import DRAW, GameResult, play_game, summary, wilson
from decks import DECKS, deck_list
from field import PANEL, panel_weights
from main import CARDS


def test_panel_lists_are_legal_and_registered_as_decks() -> None:
    assert sum(panel_weights().values()) == pytest.approx(1.0)
    for slug, archetype in PANEL.items():
        assert slug in DECKS
        validate_deck([i for i, n in archetype["cards"].items() for _ in range(n)], CARDS)


def test_field_summary_pools_and_weights_archetypes() -> None:
    def game(deck: str, won: bool) -> GameResult:
        return GameResult("field", 0, 0 if won else 1, 10, 5, 1.0, 1.0, 0.1, [0.0, 0.0], [], deck)

    results = [game("Thwackey / Dipplin", True)] * 3 + [game("Thwackey / Dipplin", False)]
    results += [game("Mega Lucario ex / Throh", False)] * 2
    weights = {"Thwackey / Dipplin": 54, "Mega Lucario ex / Throh": 53, "Unplayed": 500}
    field = summary(results, weights)["field"]
    assert isinstance(field, dict)
    assert (field["wins"], field["losses"], field["games"]) == (3, 3, 6)
    assert field["win_rate"] == 0.5
    assert field["weighted_win_rate"] == pytest.approx(0.75 * 54 / 107)
    low, high = field["wilson_95"]
    assert 0.0 <= low < 0.5 < high <= 1.0
    unweighted = summary(results)["field"]
    assert isinstance(unweighted, dict) and "weighted_win_rate" not in unweighted
    assert wilson(0, 1) == [0.0, pytest.approx(0.7934, abs=1e-3)]


def test_deck_file_can_be_overridden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    alt = tmp_path / "alt.csv"
    alt.write_text("\n".join(str(i) for i in load_deck(ROOT, CARDS)) + "\n")
    monkeypatch.setitem(os.environ, "PTCG_DECK_FILE", str(alt))
    assert load_deck(tmp_path, CARDS) == load_deck(ROOT, CARDS)


def test_games_past_the_turn_cap_are_scored_as_draws() -> None:
    lists = {"Thwackey / Dipplin": deck_list("thwackey-dipplin")}
    result = play_game(
        ("field", 0, 0, "Thwackey / Dipplin"), deck=deck_list("abomasnow"), lists=lists, max_turns=0
    )
    assert result.winner == DRAW
    assert result.turns == 1
    field = summary([result], {"Thwackey / Dipplin": 54})["field"]
    assert isinstance(field, dict)
    assert {key: field[key] for key in ("games", "wins", "draws", "losses", "win_rate")} == {
        "games": 1,
        "wins": 0,
        "draws": 1,
        "losses": 0,
        "win_rate": 0.0,
    }
    assert field["weighted_win_rate"] == 0.0
