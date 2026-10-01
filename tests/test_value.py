import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from engine import Battle, battle_finish, battle_select, battle_start
from main import ATTACKS, CARDS, POLICY
from schema import Current, Observation
from search import TERMINAL, VALUE_SCALE, Searcher
from value import FEATURE_NAMES, SIDE_FEATURES, Featurizer, ValueModel, Weights, load_model


def midgame() -> Current:
    observation: Observation = battle_start(POLICY.deck, POLICY.deck)[0]
    try:
        while True:
            selection, current = observation["select"], observation["current"]
            assert selection is not None and current is not None
            if current["result"] >= 0 or current["turn"] >= 4:
                return copy.deepcopy(current)
            observation = battle_select(POLICY.choose(observation))
    finally:
        battle_finish()
        Battle.battle_ptr = None


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
