import copy
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from engine import Battle, battle_finish, battle_select, battle_start
from main import ATTACKS, CARDS, POLICY
from schema import Current, Observation
from search import TERMINAL, VALUE_SCALE, Searcher
from value import FEATURE_NAMES, SIDE_FEATURES, Featurizer, ValueModel, Weights, load_model


def midgame() -> Current:
    """A live (non-terminal) state from turn 4 or later; replays games that end early."""
    for _ in range(20):
        observation: Observation = battle_start(POLICY.deck, POLICY.deck)[0]
        try:
            while True:
                selection, current = observation["select"], observation["current"]
                assert selection is not None and current is not None
                if current["result"] >= 0:
                    break
                if current["turn"] >= 4:
                    return copy.deepcopy(current)
                observation = battle_select(POLICY.choose(observation))
        finally:
            battle_finish()
            Battle.battle_ptr = None
    raise AssertionError("no game reached turn 4 alive")


def constant_weights(logit: float) -> Weights:
    size = len(FEATURE_NAMES)
    return {
        "features": list(FEATURE_NAMES),
        "mean": [0.0] * size,
        "scale": [1.0] * size,
        "layers": [[[0.0] * size]],
        "biases": [[logit]],
        "metadata": {},
    }


def test_features_are_fixed_length_and_mirror_between_players() -> None:
    current = midgame()
    featurizer = Featurizer(CARDS, ATTACKS)
    mine, theirs = featurizer.featurize(current, 0), featurizer.featurize(current, 1)
    assert len(mine) == len(theirs) == len(FEATURE_NAMES)
    assert all(-1.0 <= value <= 2.0 for value in mine + theirs)
    side = len(SIDE_FEATURES)
    assert mine[:side] == theirs[side : 2 * side]
    assert mine[side : 2 * side] == theirs[:side]
    assert mine[-1] == -theirs[-1]
    assert mine[FEATURE_NAMES.index("to_move")] == float(current["yourIndex"] == 0)


def test_status_flags_and_knockout_feature_follow_the_state() -> None:
    current = midgame()
    featurizer = Featurizer(CARDS, ATTACKS)
    me = current["yourIndex"]
    current["players"][me]["asleep"] = True
    assert featurizer.featurize(current, me)[FEATURE_NAMES.index("my_asleep")] == 1.0
    assert featurizer.featurize(current, 1 - me)[FEATURE_NAMES.index("their_asleep")] == 1.0
    current["players"][me]["active"] = [
        {"id": 723, "serial": 1, "playerIndex": me, "hp": 350, "maxHp": 350, "energies": [3, 3, 3]}
    ]
    current["players"][1 - me]["active"] = [
        {"id": 722, "serial": 2, "playerIndex": 1 - me, "hp": 90, "maxHp": 90}
    ]
    features = featurizer.featurize(current, me)
    assert features[FEATURE_NAMES.index("my_active_attack_ready")] == 1.0
    assert features[FEATURE_NAMES.index("my_active_can_knock_out")] == 1.0
    assert features[FEATURE_NAMES.index("their_active_attack_ready")] == 0.0


def test_model_predicts_calibrated_probabilities() -> None:
    model = ValueModel(constant_weights(0.0))
    features = [0.0] * len(FEATURE_NAMES)
    assert model.predict(features) == pytest.approx(0.5)
    assert ValueModel(constant_weights(2.0)).predict(features) == pytest.approx(
        1 / (1 + math.exp(-2.0))
    )
    assert model.parameters == len(FEATURE_NAMES) + 1
    with pytest.raises(ValueError):
        ValueModel({**constant_weights(0.0), "features": ["bogus"]})


def test_shipped_weights_load_and_beat_the_prior(tmp_path: Path) -> None:
    assert load_model(tmp_path / "missing.json") is None
    model = load_model(Path(__file__).resolve().parents[1] / "value.json")
    assert model is not None
    metadata = json.loads((Path(__file__).resolve().parents[1] / "value.json").read_text())
    assert metadata["metadata"]["holdout_log_loss"] < metadata["metadata"]["prior_log_loss"]
    current = midgame()
    probability = model.predict(Featurizer(CARDS, ATTACKS).featurize(current, 0))
    assert 0.0 < probability < 1.0


def test_learned_evaluation_is_bounded_and_terminals_dominate() -> None:
    current = midgame()
    searcher = Searcher(POLICY, None, model=ValueModel(constant_weights(3.0)))
    observation: Observation = {"select": None, "current": current}
    value = searcher.evaluate(observation, 0)
    assert value == pytest.approx(VALUE_SCALE * (2 / (1 + math.exp(-3.0)) - 1))
    assert abs(value) < VALUE_SCALE
    won: Observation = {"select": None, "current": {**current, "result": 0}}
    assert searcher.evaluate(won, 0) == TERMINAL
    assert searcher.evaluate(won, 1) == -TERMINAL


def test_baseline_worktree_loads_the_unmodified_main_agent(tmp_path: Path) -> None:
    from baseline import ROOT, ensure_worktree, load_baseline

    path = ensure_worktree(tmp_path / "main")
    try:
        baseline = load_baseline(path)
        assert baseline.choose({"select": None, "current": None}) == POLICY.deck
        loaded = sys.modules["baseline_search"].__file__
        assert loaded is not None and Path(loaded).parent == path
        assert sys.modules["search"].__file__ == str(ROOT / "search.py")
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(path)], cwd=ROOT, check=True)


def test_opponent_panel_lists_are_legal_and_weighted(tmp_path: Path) -> None:
    from assets import ROOT, load_panel

    panel = load_panel(ROOT / "opponent_panel.json", CARDS)
    assert len(panel) >= 10
    for deck, weight in panel.values():
        assert len(deck) == 60 and weight > 0
    broken = tmp_path / "panel.json"
    broken.write_text(json.dumps({"x": {"entries": 1, "cards": {"999999 Nothing": 60}}}))
    with pytest.raises(ValueError, match="unknown card"):
        load_panel(broken, CARDS)


def test_training_jobs_mix_mirror_and_panel_games_on_both_seats() -> None:
    from train_value import MIRROR, Job, generate

    panel = {"a": ([1] * 60, 3), "b": ([2] * 60, 1)}
    with patch("train_value.ProcessPoolExecutor") as executor:
        executor.return_value.__enter__.return_value.map = lambda _, jobs, chunksize: list(jobs)
        jobs = cast(list[Job], generate(400, 1, 3, panel, 0.25))
    assert len(jobs) == 400
    assert {job["our_seat"] for job in jobs} == {0, 1}
    mirrors = [job for job in jobs if job["opponent"] == MIRROR]
    assert 60 <= len(mirrors) <= 140 and all(job["opponent_deck"] is None for job in mirrors)
    others = [job for job in jobs if job["opponent"] != MIRROR]
    assert {job["opponent"] for job in others} == {"a", "b"}
    assert all(job["opponent_deck"] == panel[job["opponent"]][0] for job in others)
    counts = Counter(job["opponent"] for job in others)
    assert counts["a"] > counts["b"]


def test_vectorised_prediction_matches_the_runtime_model() -> None:
    import numpy as np

    from train_value import predict

    weights = json.loads((Path(__file__).resolve().parents[1] / "value.json").read_text())
    current = midgame()
    features = [Featurizer(CARDS, ATTACKS).featurize(current, me) for me in (0, 1)]
    batch = predict(weights, np.array(features))
    model = ValueModel(weights)
    for row, probability in zip(features, batch, strict=True):
        assert probability == pytest.approx(model.predict(row), abs=1e-9)
