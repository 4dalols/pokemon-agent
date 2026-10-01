import copy
import json
import os
import subprocess
import sys
import time
from typing import cast

import pytest

from assets import ROOT
from engine import Battle, battle_finish, battle_select, battle_start
from imitation import BCPolicy, Featurizer, Model, load_model, selection_history
from main import ATTACKS, CARDS, POLICY
from policy import Policy
from schema import Current, Observation, Selection
from search import Searcher, load_engine


def toy_model() -> Model:
    return {"weights": {}, "bias": {}}


def main_observation() -> Observation:
    observation = cast(Observation, battle_start(POLICY.deck, POLICY.deck)[0])
    try:
        while True:
            selection, current = observation["select"], observation["current"]
            assert selection is not None and current is not None
            if current["result"] >= 0:
                pytest.fail("no main selection reached")
            if selection["type"] == 0 and current["turn"] >= 2 and len(selection["option"]) > 2:
                return copy.deepcopy(observation)
            observation = cast(Observation, battle_select(POLICY.choose(observation)))
    finally:
        battle_finish()
        Battle.battle_ptr = None


@pytest.fixture(scope="module")
def observation() -> Observation:
    return main_observation()


def test_features_describe_every_option_with_kind_prefix(observation: Observation) -> None:
    selection, current = observation["select"], observation["current"]
    assert selection is not None and current is not None
    rows = Featurizer(CARDS, ATTACKS).features(selection, current, ["0:0:13:7"])
    assert len(rows) == len(selection["option"])
    assert all(tag.startswith("s=0|") for row in rows for tag in row)
    assert any("h:0:0:13:7" in tag for tag in rows[0])
    assert rows == Featurizer(CARDS, ATTACKS).features(selection, current, ["0:0:13:7"])


def test_selection_history_records_attacks_and_cards(observation: Observation) -> None:
    selection, current = observation["select"], observation["current"]
    assert selection is not None and current is not None
    prompt = copy.deepcopy(selection)
    hand = current["players"][current["yourIndex"]]["hand"]
    assert hand
    prompt["option"] = [
        {"type": 13, "attackId": 7},
        {"type": 3, "area": 2, "index": 0},
        {"type": 1},
    ]
    history = selection_history(prompt, current, [0, 1, 2, 99])
    assert history == [
        "0:atk7",
        f"0:{prompt['context']}:3:{hand[0]['id']}",
        f"0:{prompt['context']}:1",
    ]


def test_optional_picks_are_scored_relative_to_declining(observation: Observation) -> None:
    current = observation["current"]
    assert current is not None
    policy = BCPolicy(POLICY.deck, CARDS, ATTACKS, toy_model())
    prompt: Selection = {
        "type": 1,
        "context": 7,
        "minCount": 0,
        "maxCount": 2,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": [{"type": 3, "area": 1, "index": index} for index in range(3)],
        "deck": None,
        "contextCard": None,
        "effect": None,
    }
    rows = policy.featurizer.features(prompt, current, [])
    assert len(rows) == 4 and rows[-1][0] == "s=1|decline"
    policy.weights = {"s=1|decline": 1.0}
    assert policy.choose({"select": prompt, "current": current}) == []
    policy.weights = {"s=1|decline": 1.0, "s=1|pos=1": 2.0}
    assert policy.choose({"select": prompt, "current": current}) == [1]


def test_bc_policy_prefers_positively_weighted_option(observation: Observation) -> None:
    selection, current = observation["select"], observation["current"]
    assert selection is not None and current is not None
    policy = BCPolicy(POLICY.deck, CARDS, ATTACKS, toy_model())
    rows = policy.featurizer.features(selection, current, [])
    for position in range(len(selection["option"])):
        policy.weights = {rows[position][0]: 1.0}
        policy.weights[f"s=0|pos={position}"] = 5.0
        assert policy.choose(observation) == [position]
        policy.sync(current)


def test_same_turn_history_resets_when_the_turn_changes(observation: Observation) -> None:
    current = observation["current"]
    assert current is not None
    policy = BCPolicy(POLICY.deck, CARDS, ATTACKS, toy_model())
    first = policy.choose(observation)
    assert policy.history and policy.turn_key == (current["turn"], current["yourIndex"])
    state = policy.snapshot()
    policy.choose(observation)
    assert len(policy.history) == 2
    policy.restore(state)
    assert len(policy.history) == 1
    later = copy.deepcopy(observation)
    assert later["current"] is not None
    later["current"]["turn"] += 1
    assert policy.choose(later) == first
    assert len(policy.history) == 1


def test_base_policy_hooks_are_inert(observation: Observation) -> None:
    policy = Policy(POLICY.deck, CARDS, ATTACKS)
    state = policy.snapshot()
    policy.observe(observation, [0])
    policy.restore(state)
    assert policy.snapshot() == ((-1, -1), [])


def test_shipped_model_loads_and_scores_in_under_five_ms(observation: Observation) -> None:
    model = load_model(ROOT)
    assert model is not None and len(model["weights"]) > 1000
    policy = BCPolicy(POLICY.deck, CARDS, ATTACKS, model)
    selection, current = observation["select"], observation["current"]
    assert selection is not None and current is not None
    policy.choose(observation)
    started = time.perf_counter()
    for _ in range(20):
        action = policy.choose(observation)
    elapsed = (time.perf_counter() - started) / 20 * 1000
    assert len(action) == 1 and 0 <= action[0] < len(selection["option"])
    assert elapsed < 5.0, f"{elapsed:.2f} ms per decision"


def test_searcher_only_commits_the_played_action_to_history(observation: Observation) -> None:
    engine = load_engine()
    assert engine is not None
    policy = BCPolicy(POLICY.deck, CARDS, ATTACKS, toy_model())
    searcher = Searcher(policy, engine, budget=0.2, candidates=3, seed=1)
    action = searcher.choose(observation)
    selection, current = observation["select"], observation["current"]
    assert selection is not None and current is not None
    assert searcher.samples > 0
    assert len(action) == 1 and 0 <= action[0] < len(selection["option"])
    assert policy.history == selection_history(selection, current, action)


def test_bc_agent_plays_a_full_game_legally() -> None:
    model = load_model(ROOT)
    assert model is not None
    policy = BCPolicy(POLICY.deck, CARDS, ATTACKS, model)
    observation = cast(Observation, battle_start(policy.deck, policy.deck)[0])
    try:
        for _ in range(3000):
            selection, current = observation["select"], observation["current"]
            assert selection is not None and current is not None
            if current["result"] >= 0:
                break
            action = policy.choose(observation)
            assert selection["minCount"] <= len(action) <= selection["maxCount"]
            assert len(set(action)) == len(action)
            assert all(0 <= index < len(selection["option"]) for index in action)
            observation = cast(Observation, battle_select(action))
        else:
            pytest.fail("game did not finish")
    finally:
        battle_finish()
        Battle.battle_ptr = None


def test_main_wires_the_learned_policy_into_search() -> None:
    script = (
        "import json, main; from imitation import BCPolicy; "
        "assert isinstance(main.POLICY, BCPolicy) and main.SEARCHER.policy is main.POLICY; "
        "print(json.dumps(main.agent({'select': None, 'current': None})))"
    )
    env = {key: value for key, value in os.environ.items() if key != "PTCG_HEURISTIC"}
    output = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert output.returncode == 0, output.stderr
    assert json.loads(output.stdout.strip().splitlines()[-1]) == POLICY.deck


def test_selection_without_state_falls_back_to_counts() -> None:
    policy = BCPolicy(POLICY.deck, CARDS, ATTACKS, toy_model())
    prompt: Selection = {
        "type": 1,
        "context": 7,
        "minCount": 2,
        "maxCount": 2,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": [{"type": 3, "area": 6, "index": index} for index in range(4)],
        "deck": None,
        "contextCard": None,
        "effect": None,
    }
    assert policy.choose({"select": prompt, "current": None}) == [0, 1]
    assert policy.choose({"select": None, "current": cast(Current, None)}) == policy.deck
