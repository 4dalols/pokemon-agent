"""Train the learned board evaluation from self-play on the native engine.

Games are played in worker processes with the heuristic policy, short-budget rollout
search and the benchmark opponents; every MAIN selection is featurized from both
players' points of view and labelled with the final result. Our deck (`deck.csv`) sits
in one seat; the other seat plays either the same deck (mirror) or a list sampled from
an opponent panel (`--opponent-decks`, weighted by ladder `entries`), piloted by the
same policy mix. A small MLP is fitted with numpy (Adam, early stopping on games held
out by game id) and written as `value.json` for the pure-Python `value.ValueModel`;
holdout metrics are reported overall, per opponent archetype and against a previous
model (`--compare`) on the same holdout.
"""

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from assets import ROOT, load_panel
from benchmark import opponent_action
from decks import DECKS, deck_list
from engine import Battle, battle_finish, battle_select, battle_start
from main import ATTACKS, CARDS, POLICY
from policy import Policy
from schema import Observation
from search import Searcher, load_engine
from value import FEATURE_NAMES, Featurizer, Weights

OUR = POLICY

Array = NDArray[np.float64]
KINDS = ("policy", "policy", "policy", "policy", "search", "search", "greedy", "random", "first")
SEARCH_BUDGET = 0.05
STALL_LIMIT = 3000  # selections; a game still running (e.g. both players stalling) counts as a draw
MIRROR = "mirror"


class GameData(TypedDict):
    opponent: str
    features: list[list[float]]
    heuristic: list[float]
    labels: list[float]


class Job(TypedDict):
    seed: int
    kinds: tuple[str, str]
    our_seat: int
    opponent: str
    opponent_deck: list[int] | None


def play(job: Job) -> GameData:
    seed = job["seed"]
    rng = random.Random(seed)
    featurizer = Featurizer(CARDS, ATTACKS)
    engine = load_engine()
    policies = [OUR, OUR]
    if job["opponent_deck"] is not None:
        policies[1 - job["our_seat"]] = Policy(job["opponent_deck"], CARDS, ATTACKS)
    searchers = [
        Searcher(policy, engine, budget=SEARCH_BUDGET, candidates=4, seed=seed + side)
        for side, policy in enumerate(policies)
    ]
    reference = Searcher(OUR, None)
    kinds = job["kinds"]
    data: GameData = {"opponent": job["opponent"], "features": [], "heuristic": [], "labels": []}
    sides: list[int] = []
    observation, start = battle_start(policies[0].deck, policies[1].deck)
    if start.errorPlayer >= 0:
        raise ValueError(f"Native deck error: {start.errorPlayer}/{start.errorType}")
    try:
        for _ in range(STALL_LIMIT):
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
            mover = current["yourIndex"]
            kind = kinds[mover]
            if kind == "policy":
                action = policies[mover].choose(obs)
            elif kind == "search":
                action = searchers[mover].choose(obs)
            else:
                action = opponent_action(obs, kind, rng, policies[mover])
            observation = battle_select(action)
        data["labels"] = [0.5 for _ in sides]
        return data
    finally:
        battle_finish()
        Battle.battle_ptr = None


def generate(
    games: int,
    workers: int,
    seed: int,
    panel: dict[str, tuple[list[int], int]],
    mirror_share: float,
) -> list[GameData]:
    rng = random.Random(seed)
    names = list(panel)
    weights = [float(panel[name][1]) for name in names]
    jobs: list[Job] = []
    for index in range(games):
        kind = KINDS[index % len(KINDS)]
        our_seat = index % 2
        pair = (kind, "policy") if our_seat else ("policy", kind)
        opponent = MIRROR
        if names and rng.random() >= mirror_share:
            opponent = rng.choices(names, weights)[0]
        jobs.append(
            {
                "seed": seed * 1_000_003 + index,
                "kinds": pair,
                "our_seat": our_seat,
                "opponent": opponent,
                "opponent_deck": None if opponent == MIRROR else panel[opponent][0],
            }
        )
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


def predict(weights: Weights, features: Array) -> Array:
    """Vectorised `value.ValueModel.predict` for a saved weight file."""
    activations = (features - np.array(weights["mean"])) / np.array(weights["scale"])
    last = len(weights["layers"]) - 1
    for depth, (matrix, bias) in enumerate(zip(weights["layers"], weights["biases"], strict=True)):
        activations = activations @ np.array(matrix).T + np.array(bias)
        if depth < last:
            activations = np.maximum(activations, 0)
    logits = activations[:, 0]
    return np.asarray(1 / (1 + np.exp(-np.clip(logits, -30, 30))))


def evaluate(
    games: list[GameData], models: dict[str, Weights]
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Holdout log-loss/accuracy per model, overall and per opponent archetype."""
    groups = {"all": games}
    for game in games:
        groups.setdefault(game["opponent"], []).append(game)
    report: dict[str, dict[str, dict[str, float | int]]] = {}
    for group, members in groups.items():
        features, _, labels = stack(members)
        report[group] = {"games": {"count": len(members), "positions": int(len(labels))}}
        for name, weights in models.items():
            probabilities = predict(weights, features)
            report[group][name] = {
                "log_loss": log_loss(probabilities, labels),
                "accuracy": accuracy(probabilities, labels),
            }
    return report


def table(report: dict[str, dict[str, dict[str, float | int]]], models: list[str]) -> str:
    header = "| opponent | games | positions | " + " | ".join(
        f"{name} log-loss | {name} acc" for name in models
    )
    lines = [header + " |", "|" + "---|" * (3 + 2 * len(models))]
    for group, metrics in report.items():
        cells = [group, str(metrics["games"]["count"]), str(metrics["games"]["positions"])]
        for name in models:
            cells += [f"{metrics[name]['log_loss']:.4f}", f"{metrics[name]['accuracy']:.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


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
    parser.add_argument(
        "--opponent-decks",
        type=Path,
        default=ROOT / "opponent_panel.json",
        help="Panel JSON of opponent lists sampled by their `entries` weight",
    )
    parser.add_argument(
        "--mirror-share",
        type=float,
        default=0.25,
        help="Fraction of games where the opponent plays our own deck",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Earlier weight file scored on the same holdout (default: the existing --output)",
    )
    parser.add_argument("--report", type=Path, help="Write the holdout table as JSON")
    parser.add_argument(
        "--deck", choices=sorted(DECKS), help="Train for a decks.py candidate instead of deck.csv"
    )
    args = parser.parse_args()
    if args.deck is not None:
        global OUR
        OUR = Policy(deck_list(args.deck), CARDS, ATTACKS)
    started = time.perf_counter()
    compare = args.compare if args.compare is not None else args.output
    previous = cast(Weights, json.loads(compare.read_text())) if compare.exists() else None
    if args.dataset and args.dataset.exists():
        games = cast(list[GameData], json.loads(args.dataset.read_text()))
    else:
        panel = load_panel(args.opponent_decks, CARDS) if args.opponent_decks.exists() else {}
        games = generate(args.games, args.workers, args.seed, panel, args.mirror_share)
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
    opponents = Counter(game["opponent"] for game in games)
    metadata: dict[str, float | int | str] = {
        **metrics,
        "games": len(games),
        "positions": int(len(train_y) + len(held_y)),
        "hidden": args.hidden,
        "seed": args.seed,
        "generation_seconds": round(generation, 1),
        "mirror_games": opponents[MIRROR],
        "opponent_decks": len(opponents) - (MIRROR in opponents),
    }
    weights = export(network, mean, scale, metadata)
    models = {"new": weights}
    if previous is not None:
        models["old"] = previous
        held = evaluate(games[split:], models)
        metadata["compare"] = str(compare)
        metadata["compare_log_loss"] = held["all"]["old"]["log_loss"]
        metadata["compare_accuracy"] = held["all"]["old"]["accuracy"]
    else:
        held = evaluate(games[split:], models)
    args.output.write_text(json.dumps(weights))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"metadata": metadata, "holdout": held}, indent=2))
    print(json.dumps(metadata, indent=2))
    print(table(held, list(models)))


if __name__ == "__main__":
    main()
