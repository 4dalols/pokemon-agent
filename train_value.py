"""Train the learned board evaluation from self-play on the native engine.

Games are played in worker processes with the heuristic policy, short-budget rollout
search and the benchmark opponents; every MAIN selection is featurized from both
players' points of view and labelled with the final result. A small MLP is fitted
with numpy (Adam, early stopping on games held out by game id) and written as
`value.json` for the pure-Python `value.ValueModel`.
"""

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from benchmark import opponent_action
from engine import Battle, battle_finish, battle_select, battle_start
from main import ATTACKS, CARDS, POLICY
from schema import Observation
from search import Searcher, load_engine
from value import FEATURE_NAMES, Featurizer, Weights

Array = NDArray[np.float64]
KINDS = ("policy", "policy", "policy", "policy", "search", "search", "greedy", "random", "first")
SEARCH_BUDGET = 0.05


class GameData(TypedDict):
    features: list[list[float]]
    heuristic: list[float]
    labels: list[float]


def play(job: tuple[int, str, str]) -> GameData:
    seed, left, right = job
    rng = random.Random(seed)
    featurizer = Featurizer(CARDS, ATTACKS)
    searcher = Searcher(POLICY, load_engine(), budget=SEARCH_BUDGET, candidates=4, seed=seed)
    reference = Searcher(POLICY, None)
    kinds = (left, right)
    data: GameData = {"features": [], "heuristic": [], "labels": []}
    sides: list[int] = []
    observation, start = battle_start(POLICY.deck, POLICY.deck)
    if start.errorPlayer >= 0:
        raise ValueError(f"Native deck error: {start.errorPlayer}/{start.errorType}")
    try:
        for _ in range(10000):
            obs = cast(Observation, observation)
            current, selection = obs["current"], obs["select"]
            if current is None or selection is None:
                raise ValueError("Missing game state")
            if current["result"] >= 0:
                result = current["result"]
                data["labels"] = [
                    1.0 if result == me else 0.0 if result == 1 - me else 0.5 for me in sides
                ]
                return data
            if selection["type"] == 0:
                for me in (0, 1):
                    data["features"].append(featurizer.featurize(current, me))
                    data["heuristic"].append(reference.evaluate(obs, me))
                    sides.append(me)
            kind = kinds[current["yourIndex"]]
            if kind == "policy":
                action = POLICY.choose(obs)
            elif kind == "search":
                action = searcher.choose(obs)
            else:
                action = opponent_action(obs, kind, rng)
            observation = battle_select(action)
        raise RuntimeError("Game exceeded 10,000 selections")
    finally:
        battle_finish()
        Battle.battle_ptr = None


def generate(games: int, workers: int, seed: int) -> list[GameData]:
    rng = random.Random(seed)
    jobs = []
    for index in range(games):
        kind = KINDS[index % len(KINDS)]
        pair = ("policy", kind) if index % 2 else (kind, "policy")
        jobs.append((seed * 1_000_003 + index, *pair))
    rng.shuffle(jobs)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(play, jobs, chunksize=8))


def stack(games: list[GameData]) -> tuple[Array, Array, Array]:
    features = np.array([row for game in games for row in game["features"]], dtype=np.float64)
    heuristic = np.array([h for game in games for h in game["heuristic"]], dtype=np.float64)
    labels = np.array([y for game in games for y in game["labels"]], dtype=np.float64)
    return features, heuristic, labels


def log_loss(probabilities: Array, labels: Array) -> float:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def accuracy(probabilities: Array, labels: Array) -> float:
    decided = labels != 0.5
    return float(np.mean((probabilities[decided] > 0.5) == (labels[decided] > 0.5)))


class Network:
    def __init__(self, sizes: list[int], rng: np.random.Generator) -> None:
        self.weights = [
            rng.normal(0, (2 / fan_in) ** 0.5, size=(fan_out, fan_in))
            for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True)
        ]
        self.biases = [np.zeros(fan_out) for fan_out in sizes[1:]]

    def forward(self, inputs: Array) -> tuple[Array, list[Array]]:
        activations = [inputs]
        for depth, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            outputs = activations[-1] @ weight.T + bias
            if depth < len(self.weights) - 1:
                outputs = np.maximum(outputs, 0)
            activations.append(outputs)
        logits = activations[-1][:, 0]
        return 1 / (1 + np.exp(-np.clip(logits, -30, 30))), activations

    def gradients(self, activations: list[Array], delta: Array) -> list[tuple[Array, Array]]:
        grads: list[tuple[Array, Array]] = []
        signal = delta[:, None]
        for depth in range(len(self.weights) - 1, -1, -1):
            inputs = activations[depth]
            grads.append((signal.T @ inputs / len(inputs), signal.mean(axis=0)))
            if depth > 0:
                signal = (signal @ self.weights[depth]) * (inputs > 0)
        return grads[::-1]


def train(
    inputs: Array,
    labels: Array,
    holdout: tuple[Array, Array],
    hidden: int,
    epochs: int,
    seed: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch: int = 256,
) -> tuple[Network, dict[str, float]]:
    rng = np.random.default_rng(seed)
    sizes = [inputs.shape[1], hidden, 1] if hidden > 0 else [inputs.shape[1], 1]
    network = Network(sizes, rng)
    params = network.weights + network.biases
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    best_loss = float("inf")
    best = [p.copy() for p in params]
    step = 0
    patience = 0
    for epoch in range(epochs):
        order = rng.permutation(len(inputs))
        for start in range(0, len(order), batch):
            index = order[start : start + batch]
            probabilities, activations = network.forward(inputs[index])
            grads = network.gradients(activations, probabilities - labels[index])
            flat = [g for g, _ in grads] + [b for _, b in grads]
            step += 1
            for position, (param, grad) in enumerate(zip(params, flat, strict=True)):
                if position < len(network.weights):
                    grad = grad + weight_decay * param
                first[position] = 0.9 * first[position] + 0.1 * grad
                second[position] = 0.999 * second[position] + 0.001 * grad * grad
                m_hat = first[position] / (1 - 0.9**step)
                v_hat = second[position] / (1 - 0.999**step)
                param -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        held_probabilities, _ = network.forward(holdout[0])
        loss = log_loss(held_probabilities, holdout[1])
        if loss < best_loss - 1e-5:
            best_loss, patience = loss, 0
            best = [p.copy() for p in params]
        else:
            patience += 1
            if patience >= 5:
                break
        print(f"epoch {epoch + 1}: holdout log-loss {loss:.4f}")
    for param, value in zip(params, best, strict=True):
        param[...] = value
    probabilities, _ = network.forward(holdout[0])
    return network, {
        "holdout_log_loss": log_loss(probabilities, holdout[1]),
        "holdout_accuracy": accuracy(probabilities, holdout[1]),
    }


def export(
    network: Network, mean: Array, scale: Array, metadata: dict[str, float | int | str]
) -> Weights:
    return {
        "features": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "layers": [weight.tolist() for weight in network.weights],
        "biases": [bias.tolist() for bias in network.biases],
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("value.json"))
    parser.add_argument("--dataset", type=Path, help="Cache generated games as JSON")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.dataset and args.dataset.exists():
        games = cast(list[GameData], json.loads(args.dataset.read_text()))
    else:
        games = generate(args.games, args.workers, args.seed)
        if args.dataset:
            args.dataset.parent.mkdir(parents=True, exist_ok=True)
            args.dataset.write_text(json.dumps(games))
    generation = time.perf_counter() - started
    split = int(len(games) * (1 - args.holdout))
    train_x, _, train_y = stack(games[:split])
    held_x, held_h, held_y = stack(games[split:])
    mean = train_x.mean(axis=0)
    scale = np.where(train_x.std(axis=0) > 1e-6, train_x.std(axis=0), 1.0)
    network, metrics = train(
        (train_x - mean) / scale,
        train_y,
        ((held_x - mean) / scale, held_y),
        args.hidden,
        args.epochs,
        args.seed,
    )
    prior = np.full(len(held_y), train_y.mean())
    metrics["prior_log_loss"] = log_loss(prior, held_y)
    metrics["heuristic_accuracy"] = accuracy(1 / (1 + np.exp(-held_h / 300)), held_y)
    metadata: dict[str, float | int | str] = {
        **metrics,
        "games": len(games),
        "positions": int(len(train_y) + len(held_y)),
        "hidden": args.hidden,
        "seed": args.seed,
        "generation_seconds": round(generation, 1),
    }
    args.output.write_text(json.dumps(export(network, mean, scale, metadata)))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
