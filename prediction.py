"""Measure hidden-card prediction quality against the opponent's actual draws.

Plays our agent (policy only, to keep it fast) against a greedy opponent piloting a
chosen deck. At every one of our decisions both predictors estimate the distribution
of the opponent's next drawn card; the log-likelihood of the card the opponent then
really draws (read from the opponent's own observation logs) is accumulated.
"""

import argparse
import json
import math
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

from archetypes import LIBRARY
from benchmark import opponent_action
from engine import Battle, battle_finish, battle_select, battle_start
from main import POLICY, SEARCHER
from memory import LOG_DRAW, Tracker
from schema import Current, Observation, Player

SAMPLES = 24
SMOOTHING = 0.5


def distribution(samples: list[list[int]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for sample in samples:
        counts.update(sample)
    return counts


def likelihood(counts: Counter[int], card_id: int) -> float:
    total = sum(counts.values()) + SMOOTHING * len(POLICY.cards)
    return math.log((counts[card_id] + SMOOTHING) / total)


def baseline_deck(theirs: Player) -> list[int]:
    pool = SEARCHER.enemy_pool(theirs)
    SEARCHER.rng.shuffle(pool)
    hidden_prizes = sum(card is None for card in theirs["prize"])
    return pool[hidden_prizes + theirs["handCount"] :]


def tracked_deck(tracker: Tracker, theirs: Player, current: Current) -> list[int]:
    side = tracker.opponent(current)
    sampled = SEARCHER.predictor.sample(theirs, current, side, tracker.revealed(theirs, current))
    if sampled is None:
        return baseline_deck(theirs)
    return sampled[0]


def play(job: tuple[str, int]) -> dict[str, float]:
    name, seed = job
    deck = next(list(entry.cards.elements()) for entry in LIBRARY if entry.name == name)
    rng = random.Random(seed)
    tracker = Tracker(POLICY.deck)
    SEARCHER.rng.seed(seed)
    me = seed % 2
    observation, _ = battle_start(*(POLICY.deck, deck) if me == 0 else (deck, POLICY.deck))
    totals = {"baseline": 0.0, "tracked": 0.0, "draws": 0.0, "decisions": 0.0}
    pending: tuple[Counter[int], Counter[int]] | None = None
    try:
        for _ in range(10000):
            obs = cast(Observation, observation)
            current, selection = obs["current"], obs["select"]
            if current is None or current["result"] >= 0 or selection is None:
                break
            if current["yourIndex"] == me:
                tracker.observe(obs)
                theirs = current["players"][1 - me]
                if theirs["deckCount"] > 0:
                    pending = (
                        distribution([baseline_deck(theirs) for _ in range(SAMPLES)]),
                        distribution(
                            [tracked_deck(tracker, theirs, current) for _ in range(SAMPLES)]
                        ),
                    )
                    totals["decisions"] += 1
                action = POLICY.choose(obs)
            else:
                for log in obs.get("logs") or []:
                    if log.get("type") == LOG_DRAW and log.get("playerIndex") == 1 - me:
                        card_id = log.get("cardId")
                        if pending is not None and isinstance(card_id, int):
                            totals["baseline"] += likelihood(pending[0], card_id)
                            totals["tracked"] += likelihood(pending[1], card_id)
                            totals["draws"] += 1
                action = opponent_action(obs, "greedy", rng)
            observation = battle_select(action)
    finally:
        battle_finish()
        Battle.battle_ptr = None
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=8, help="Games per opponent deck")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--decks", nargs="+", default=[e.name for e in LIBRARY[:8]])
    parser.add_argument("--output", type=Path, default=Path("results/prediction.json"))
    args = parser.parse_args()
    jobs = [(name, i) for name in args.decks for i in range(args.games)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(play, jobs))
    report: dict[str, dict[str, float]] = {}
    for (name, _), result in zip(jobs, results, strict=True):
        entry = report.setdefault(name, {"baseline": 0.0, "tracked": 0.0, "draws": 0.0})
        for key in entry:
            entry[key] += result[key]
    for entry in report.values():
        draws = max(1.0, entry["draws"])
        entry["baseline_per_draw"] = entry["baseline"] / draws
        entry["tracked_per_draw"] = entry["tracked"] / draws
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
