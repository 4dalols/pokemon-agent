"""Train the imitation ranker from extracted Playground decisions.

Input: a directory of per-episode pickles produced by `tools/extract_replays.py`
(one list of decision dicts per episode). Episodes are split into train/held-out by
hash; the held-out top-1 accuracy per selection type is reported next to the
heuristic `Policy` on the same decisions. Output: `bc_model.json`.
"""

import argparse
import hashlib
import json
import pickle
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from assets import ROOT, load_catalog
from imitation import Featurizer, Model, selection_history
from policy import Policy
from schema import Observation

CARDS, ATTACKS = load_catalog(ROOT)
TOP_TEAMS = {
    "No Free MIST Energy",
    "Hinata Tokuda",
    "Raihan Ramadistra",
    "Belati Jagad Bintang Syuhada",
    "Leon Liu",
    "hwe owe",
    "Shardul Gharat",
    "C4",
    "Steve421471",
}


class Decision(TypedDict):
    episode: int
    agent: int
    team: str
    reward: int
    deck: list[int]
    dragapult: bool
    avg_score: float
    min_score: float
    step: int
    observation: Observation
    action: list[int]


class Example(TypedDict):
    key: str
    kind: int
    max_count: int
    options: list[list[str]]
    chosen: list[int]
    heuristic: list[int]
    weight: float
    held_out: bool


def held_out(episode: int) -> bool:
    return int(hashlib.sha1(str(episode).encode()).hexdigest(), 16) % 5 == 0


def featurize_file(args: tuple[Path, float, float]) -> list[Example]:
    path, min_score, dragapult_weight = args
    rows = cast(list[Decision], pickle.load(path.open("rb")))
    featurizer = Featurizer(CARDS, ATTACKS)
    examples: list[Example] = []
    streams: dict[int, list[Decision]] = defaultdict(list)
    for row in rows:
        if row["team"] in TOP_TEAMS or row["avg_score"] >= min_score:
            streams[row["agent"]].append(row)
    for stream in streams.values():
        stream.sort(key=lambda row: row["step"])
        policy = Policy(stream[0]["deck"], CARDS, ATTACKS)
        history: list[str] = []
        turn_key = (-1, -1)
        for row in stream:
            observation = row["observation"]
            selection, current = observation["select"], observation["current"]
            if selection is None or current is None:
                continue
            key = (current["turn"], current["yourIndex"])
            if key != turn_key:
                turn_key, history = key, []
            options = selection["option"]
            action = [index for index in row["action"] if 0 <= index < len(options)]
            rows_ = featurizer.features(selection, current, history)
            decline = [len(options)] if selection["minCount"] == 0 else []
            chosen = action or decline
            if len(rows_) >= 2 and chosen:
                examples.append(
                    {
                        "key": f"{row['episode']}:{row['agent']}",
                        "kind": selection["type"],
                        "max_count": selection["maxCount"],
                        "options": rows_,
                        "chosen": chosen,
                        "heuristic": policy.choose(observation) or decline,
                        "weight": dragapult_weight if row["dragapult"] else 1.0,
                        "held_out": held_out(row["episode"]),
                    }
                )
            history.extend(selection_history(selection, current, action))
            del history[:-12]
    return examples


def build_vocab(examples: list[Example], min_count: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in examples:
        for row in example["options"]:
            counts.update(row)
    vocab = {"": 0}
    for tag, count in counts.items():
        if count >= min_count:
            vocab[tag] = len(vocab)
    return vocab


class Batch:
    def __init__(self, examples: list[Example], vocab: dict[str, int], width: int) -> None:
        offsets = [0]
        features: list[NDArray[np.int32]] = []
        labels: list[float] = []
        weights: list[float] = []
        kinds: list[int] = []
        listwise: list[bool] = []
        for example in examples:
            rows = example["options"]
            chosen = set(example["chosen"])
            for position, row in enumerate(rows):
                ids = [vocab[tag] for tag in row if tag in vocab][:width]
                ids.extend([0] * (width - len(ids)))
                features.append(np.asarray(ids, dtype=np.int32))
                labels.append(float(position in chosen))
            offsets.append(len(features))
            weights.append(example["weight"])
            kinds.append(example["kind"])
            listwise.append(example["max_count"] == 1 and len(chosen) == 1)
        self.features: NDArray[np.int32] = (
            np.stack(features) if features else np.zeros((0, width), dtype=np.int32)
        )
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.labels = np.asarray(labels)
        self.weights = np.asarray(weights)
        self.kinds = np.asarray(kinds)
        self.listwise = np.asarray(listwise)
        self.groups = np.repeat(np.arange(len(examples)), np.diff(self.offsets))

    def __len__(self) -> int:
        return len(self.weights)


def scores_of(
    batch: Batch, weights: NDArray[np.float64], bias: NDArray[np.float64]
) -> NDArray[np.float64]:
    scores: NDArray[np.float64] = weights[batch.features].sum(axis=1)
    offsets: NDArray[np.float64] = bias[batch.kinds][batch.groups]
    return scores + offsets


def grad_step(
    batch: Batch, weights: NDArray[np.float64], bias: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    scores = scores_of(batch, weights, bias)
    starts = batch.offsets[:-1]
    group_max = np.maximum.reduceat(scores, starts)
    shifted = np.exp(scores - group_max[batch.groups])
    totals = np.add.reduceat(shifted, starts)
    softmax = shifted / totals[batch.groups]
    sigmoid = 1.0 / (1.0 + np.exp(-scores))
    use_list = batch.listwise[batch.groups]
    probabilities = np.where(use_list, softmax, sigmoid)
    grad = probabilities - batch.labels
    option_weight = batch.weights[batch.groups]
    eps = 1e-9
    loss_terms = np.where(
        use_list,
        -batch.labels * np.log(probabilities + eps),
        -(batch.labels * np.log(sigmoid + eps) + (1 - batch.labels) * np.log(1 - sigmoid + eps)),
    )
    loss = float((loss_terms * option_weight).sum() / batch.weights.sum())
    grad = grad * option_weight / batch.weights.sum()
    width = batch.features.shape[1]
    grad_w = np.bincount(
        batch.features.ravel(), weights=np.repeat(grad, width), minlength=len(weights)
    ).astype(np.float64)
    grad_w[0] = 0.0
    grad_b = np.bincount(batch.kinds[batch.groups], weights=grad, minlength=len(bias)).astype(
        np.float64
    )
    return loss, grad_w, grad_b


def train(
    examples: list[Example],
    vocab: dict[str, int],
    width: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    weights: NDArray[np.float64] = np.zeros(len(vocab))
    bias: NDArray[np.float64] = np.zeros(16)
    m_w: NDArray[np.float64] = np.zeros_like(weights)
    v_w: NDArray[np.float64] = np.zeros_like(weights)
    m_b: NDArray[np.float64] = np.zeros_like(bias)
    v_b: NDArray[np.float64] = np.zeros_like(bias)
    beta1, beta2 = 0.9, 0.999
    step = 0
    batches = [
        Batch(examples[start : start + batch_size], vocab, width)
        for start in range(0, len(examples), batch_size)
    ]
    for epoch in range(epochs):
        order = rng.permutation(len(batches))
        total = 0.0
        started = time.perf_counter()
        for index in order:
            batch = batches[index]
            if len(batch) == 0:
                continue
            loss, grad_w, grad_b = grad_step(batch, weights, bias)
            grad_w += l2 * weights
            step += 1
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            scale = learning_rate * (1 - beta2**step) ** 0.5 / (1 - beta1**step)
            weights -= scale * m_w / (np.sqrt(v_w) + 1e-8)
            bias -= scale * m_b / (np.sqrt(v_b) + 1e-8)
            total += loss
        print(
            f"epoch {epoch + 1}: loss {total / max(1, len(batches)):.4f}"
            f" ({time.perf_counter() - started:.0f}s)"
        )
    return weights, bias


def evaluate(
    examples: list[Example],
    vocab: dict[str, int],
    width: int,
    weights: NDArray[np.float64],
    bias: NDArray[np.float64],
) -> dict[str, dict[str, float]]:
    hits: Counter[str] = Counter()
    heuristic_hits: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for start in range(0, len(examples), 2048):
        chunk = examples[start : start + 2048]
        batch = Batch(chunk, vocab, width)
        scores = scores_of(batch, weights, bias)
        for index, example in enumerate(chunk):
            own = scores[batch.offsets[index] : batch.offsets[index + 1]]
            if example["max_count"] != 1 or len(example["chosen"]) != 1:
                predicted = sorted(
                    range(len(own)), key=lambda position: (-own[position], position)
                )[: len(example["chosen"])]
                hit = set(predicted) == set(example["chosen"])
                heuristic_hit = set(example["heuristic"]) == set(example["chosen"])
                name = f"type{example['kind']}-multi"
            else:
                hit = int(np.argmax(own)) == example["chosen"][0]
                heuristic_hit = example["heuristic"][:1] == example["chosen"]
                name = f"type{example['kind']}"
            for label in (name, "all"):
                totals[label] += 1
                hits[label] += hit
                heuristic_hits[label] += heuristic_hit
    return {
        name: {
            "n": totals[name],
            "bc_top1": hits[name] / totals[name],
            "heuristic_top1": heuristic_hits[name] / totals[name],
        }
        for name in sorted(totals)
    }


def export(
    vocab: dict[str, int], weights: NDArray[np.float64], bias: NDArray[np.float64], path: Path
) -> None:
    model: Model = {
        "weights": {
            tag: round(float(weights[index]), 5)
            for tag, index in vocab.items()
            if index > 0 and abs(weights[index]) > 1e-4
        },
        "bias": {f"s={kind}": round(float(bias[kind]), 5) for kind in range(len(bias))},
    }
    path.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {path} with {len(model['weights']):,} weights ({path.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--min-score", type=float, default=700.0)
    parser.add_argument("--dragapult-weight", type=float, default=2.0)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "bc_model.json")
    parser.add_argument("--report", type=Path, default=ROOT / "results/bc_eval.json")
    args = parser.parse_args()
    files = sorted(args.decisions.glob("*.pkl"))
    if args.limit:
        files = files[: args.limit]
    started = time.perf_counter()
    jobs = [(path, args.min_score, args.dragapult_weight) for path in files]
    with ProcessPoolExecutor(args.workers) as pool:
        examples = [example for chunk in pool.map(featurize_file, jobs) for example in chunk]
    train_set = [example for example in examples if not example["held_out"]]
    test_set = [example for example in examples if example["held_out"]]
    print(
        f"{len(train_set):,} train / {len(test_set):,} held-out decisions"
        f" in {time.perf_counter() - started:.0f}s"
    )
    vocab = build_vocab(train_set, args.min_count)
    print(f"vocabulary {len(vocab):,}")
    weights, bias = train(
        train_set,
        vocab,
        args.width,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.l2,
        seed=0,
    )
    report = evaluate(test_set, vocab, args.width, weights, bias)
    for name, entry in report.items():
        print(
            f"{name:14s} n={int(entry['n']):6d} bc={entry['bc_top1']:.3f}"
            f" heuristic={entry['heuristic_top1']:.3f}"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    export(vocab, weights, bias, args.output)


if __name__ == "__main__":
    main()
