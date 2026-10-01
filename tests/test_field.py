import os
from pathlib import Path

import pytest

from assets import ROOT, load_deck, validate_deck
from benchmark import DRAW, GameResult, field_jobs, play_game, summary, wilson
from decks import DECKS
from field import PANEL, panel_weights
from main import CARDS


def test_panel_lists_are_legal_and_registered_as_decks() -> None:
    assert sum(panel_weights().values()) == pytest.approx(1.0)
    for slug, archetype in PANEL.items():
        assert slug in DECKS
        validate_deck([i for i, n in archetype["cards"].items() for _ in range(n)], CARDS)


def test_field_jobs_follow_entry_shares_and_alternate_seats() -> None:
    jobs = field_jobs(150, 7)
    assert len(jobs) == 150
    assert len({seed for _, _, seed in jobs}) == 150 and min(seed for _, _, seed in jobs) == 7
    counts = {slug: 0 for slug in PANEL}
    for name, _, _ in jobs:
        counts[name[6:]] += 1
    for slug, weight in panel_weights().items():
        assert abs(counts[slug] - 150 * weight) < 1
    seats = [seat for name, seat, _ in jobs if name == "field:hydrapple_ex-teal_mask_ogerpon_ex"]
    assert seats.count(0) in (len(seats) // 2, (len(seats) + 1) // 2)


def test_field_summary_pools_and_weights_archetypes() -> None:
    def game(name: str, won: bool) -> GameResult:
        return GameResult(name, 0, 0 if won else 1, 10, 5, 1.0, 1.0, 0.1, [0])

    results = [game("field:thwackey-dipplin", True)] * 3 + [game("field:thwackey-dipplin", False)]
    results += [game("field:mega_lucario_ex-throh", False)] * 2
    report = summary(results)
    field = report["field"]
    assert isinstance(field, dict)
    assert (field["wins"], field["losses"], field["games"]) == (3, 3, 6)
    assert field["win_rate"] == 0.5
    weights = panel_weights()
    share = weights["thwackey-dipplin"] / (
        weights["thwackey-dipplin"] + weights["mega_lucario_ex-throh"]
    )
    assert field["weighted_win_rate"] == pytest.approx(0.75 * share)
    low, high = field["wilson_95"]
    assert 0.0 <= low < 0.5 < high <= 1.0


def test_deck_file_can_be_overridden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    alt = tmp_path / "alt.csv"
    alt.write_text("\n".join(str(i) for i in load_deck(ROOT, CARDS)) + "\n")
    monkeypatch.setitem(os.environ, "PTCG_DECK_FILE", str(alt))
    assert load_deck(tmp_path, CARDS) == load_deck(ROOT, CARDS)


def test_games_past_the_turn_cap_are_scored_as_draws() -> None:
    result = play_game(("field:thwackey-dipplin", 0, 0), deck="abomasnow", max_turns=0)
    assert result.winner == DRAW
    assert result.turns == 1
    field = summary([result])["field"]
    assert isinstance(field, dict)
    assert {key: field[key] for key in ("games", "wins", "draws", "losses", "win_rate")} == {
        "games": 1,
        "wins": 0,
        "draws": 1,
        "losses": 0,
        "win_rate": 0.0,
    }
    assert field["wilson_95"] == wilson(0, 1)
