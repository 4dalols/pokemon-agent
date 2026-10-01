import argparse
import atexit
import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import cast

from assets import ROOT
from decks import DECKS, deck_list
from engine import SOURCE, Battle, battle_finish, battle_select, battle_start
from field import PANEL, panel_weights
from main import ATTACKS, CARDS, POLICY, SEARCHER
from policy import Policy
from reference import ReferenceAgent, ensure_reference
from schema import Observation
from search import Searcher, load_engine

MAIN_REF = ROOT / "results" / "main-ref"
SIMPLE_OPPONENTS = ("first", "random", "greedy", "self", "main", "field")

AGENTS: dict[str, tuple[Policy, Searcher]] = {}
REFERENCES: dict[tuple[Path, bool], ReferenceAgent] = {}


@dataclass
class GameResult:
    opponent: str
    baseline_seat: int
    winner: int
    steps: int
    turns: int
    max_decision_ms: float
    opponent_max_decision_ms: float
    seconds: float
    contexts: list[int]


def agent_for(deck: str | None) -> tuple[Policy, Searcher]:
    """Policy and searcher for a named candidate deck (``None`` means the shipped deck.csv)."""
    if deck is None:
        return POLICY, SEARCHER
    if deck not in AGENTS:
        policy = Policy(deck_list(deck), CARDS, ATTACKS)
        AGENTS[deck] = policy, Searcher(policy, load_engine(), SEARCHER.budget, SEARCHER.candidates)
    return AGENTS[deck]


def reference_for(root: Path, search: bool) -> ReferenceAgent:
    if (root, search) not in REFERENCES:
        REFERENCES[root, search] = ReferenceAgent(root, search=search)
        atexit.register(REFERENCES[root, search].close)
    return REFERENCES[root, search]


def opponent_action(observation: Observation, kind: str, rng: random.Random) -> list[int]:
    selection = observation["select"]
    if selection is None:
        return POLICY.deck.copy()
    options = selection["option"]
    if kind == "random":
        return rng.sample(range(len(options)), selection["maxCount"])
    if kind == "greedy" and selection["type"] == 0:
        priorities = {9: 5, 8: 4, 7: 3, 10: -1, 12: -1, 13: 2, 14: 0}
        return [
            max(
                range(len(options)),
                key=lambda i: (
                    priorities.get(options[i]["type"], -1),
                    ATTACKS[options[i]["attackId"]]["damage"] if options[i]["type"] == 13 else 0,
                    int(options[i].get("inPlayArea", 4) == 4),
                    -i,
                ),
            )
        ]
    return list(range(selection["maxCount"]))


def validate_opponent(name: str) -> str:
    if name in SIMPLE_OPPONENTS:
        return name
    if name.startswith("deck:") and name[5:] in DECKS:
        return name
    raise argparse.ArgumentTypeError(
        f"Unknown opponent {name!r}; use {', '.join(SIMPLE_OPPONENTS)} or deck:<name>"
    )


def play_game(
    job: tuple[str, int, int],
    trace_path: Path | None = None,
    search: bool = False,
    deck: str | None = None,
    opponent_search: bool = False,
    main_ref: Path = MAIN_REF,
) -> GameResult:
    opponent, seat, seed = job
    rng = random.Random(seed)
    started = time.perf_counter()
    policy, searcher = agent_for(deck)
    reference = reference_for(main_ref, opponent_search) if opponent == "main" else None
    rival: tuple[Policy, Searcher] | None = None
    if opponent.startswith("deck:"):
        rival = agent_for(opponent[5:])
    elif opponent.startswith("field:"):
        rival = agent_for(opponent[6:])
        opponent_search = opponent_search or search
    elif opponent == "self":
        rival = policy, searcher
    if reference is not None:
        opponent_deck = reference.deck()
    elif rival is not None:
        opponent_deck = rival[0].deck
    else:
        opponent_deck = policy.deck
    decks = (policy.deck, opponent_deck) if seat == 0 else (opponent_deck, policy.deck)
    observation, start = battle_start(*decks)
    if start.errorPlayer >= 0:
        raise ValueError(f"Native deck error: {start.errorPlayer}/{start.errorType}")
    contexts: set[int] = set()
    maximum = 0.0
    opponent_maximum = 0.0
    trace: list[dict[str, object]] = []
    try:
        for step in range(10000):
            obs = cast(Observation, observation)
            current, selection = obs["current"], obs["select"]
            if current is None:
                raise ValueError("Missing game state")
            if current["result"] >= 0:
                return GameResult(
                    opponent,
                    seat,
                    current["result"],
                    step,
                    current["turn"],
                    maximum,
                    opponent_maximum,
                    time.perf_counter() - started,
                    sorted(contexts),
                )
            if selection is None:
                raise ValueError("Missing live selection")
            contexts.add(selection["context"])
            decision_started = time.perf_counter()
            if current["yourIndex"] == seat:
                action = searcher.choose(obs) if search else policy.choose(obs)
                maximum = max(maximum, (time.perf_counter() - decision_started) * 1000)
            elif reference is not None:
                action = reference.choose(obs)
            elif rival is not None:
                action = rival[1].choose(obs) if opponent_search else rival[0].choose(obs)
            else:
                action = opponent_action(obs, opponent, rng)
            if current["yourIndex"] != seat:
                elapsed = (time.perf_counter() - decision_started) * 1000
                opponent_maximum = max(opponent_maximum, elapsed)
            if trace_path:
                trace.append({"observation": obs, "action": action})
            if not selection["minCount"] <= len(action) <= selection["maxCount"]:
                raise ValueError(f"Invalid count: {selection}, {action}")
            if len(set(action)) != len(action) or any(
                i < 0 or i >= len(selection["option"]) for i in action
            ):
                raise ValueError(f"Invalid option indices: {selection}, {action}")
            try:
                observation = battle_select(action)
            except IndexError as exc:
                raise ValueError(f"Engine rejected selection: {selection}, {action}") from exc
        raise RuntimeError("Game exceeded 10,000 selections")
    finally:
        battle_finish()
        Battle.battle_ptr = None
        if trace_path:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps(trace, indent=2))


def wilson(wins: int, n: int) -> list[float]:
    z = 1.96
    proportion = wins / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    margin = z * (proportion * (1 - proportion) / n + z * z / (4 * n * n)) ** 0.5 / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def field_jobs(games: int, offset: int) -> list[tuple[str, int, int]]:
    """Games against the ladder panel, allocated to archetypes in proportion to their entries."""
    weights = panel_weights()
    counts = {slug: int(games * weight) for slug, weight in weights.items()}
    remainders = sorted(weights, key=lambda slug: counts[slug] - games * weights[slug])
    for slug in remainders[: games - sum(counts.values())]:
        counts[slug] += 1
    jobs: list[tuple[str, int, int]] = []
    for slug, count in counts.items():
        base = offset + len(jobs)
        jobs.extend((f"field:{slug}", i % 2, base + i) for i in range(count))
    return jobs


def field_summary(groups: dict[str, dict[str, object]]) -> dict[str, object] | None:
    """Pooled and entry-weighted win rate over the ``field:<archetype>`` groups."""
    weights = panel_weights()
    present = {name[6:]: entry for name, entry in groups.items() if name.startswith("field:")}
    if not present:
        return None
    wins = sum(cast(int, entry["wins"]) for entry in present.values())
    games = sum(cast(int, entry["games"]) for entry in present.values())
    total_weight = sum(weights[slug] for slug in present)
    weighted = sum(
        weights[slug] * cast(float, entry["win_rate"]) / total_weight
        for slug, entry in present.items()
    )
    return {
        "games": games,
        "wins": wins,
        "losses": games - wins - sum(cast(int, entry["draws"]) for entry in present.values()),
        "win_rate": wins / games,
        "wilson_95": wilson(wins, games),
        "weighted_win_rate": weighted,
        "max_decision_ms": max(cast(float, entry["max_decision_ms"]) for entry in present.values()),
        "archetypes": {slug: PANEL[slug]["name"] for slug in present},
    }


def summary(results: list[GameResult], wall_seconds: float | None = None) -> dict[str, object]:
    groups: dict[str, dict[str, object]] = {}
    for name in sorted({result.opponent for result in results}):
        group = [result for result in results if result.opponent == name]
        wins = sum(result.winner == result.baseline_seat for result in group)
        draws = sum(result.winner not in (0, 1) for result in group)
        n = len(group)
        entry: dict[str, object] = {
            "games": n,
            "wins": wins,
            "draws": draws,
            "losses": n - wins - draws,
            "win_rate": wins / n,
            "wilson_95": wilson(wins, n),
            "max_decision_ms": max(result.max_decision_ms for result in group),
            "opponent_max_decision_ms": max(result.opponent_max_decision_ms for result in group),
            "mean_game_seconds": sum(result.seconds for result in group) / n,
            "mean_turns": sum(result.turns for result in group) / n,
            "seats": dict(Counter(result.baseline_seat for result in group)),
        }
        if wall_seconds:
            entry["games_per_second"] = len(results) / wall_seconds
        groups[name] = entry
    field = field_summary(groups)
    report: dict[str, object] = dict(groups)
    if field is not None:
        report["field"] = field
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=40, help="Games per opponent")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--opponents",
        nargs="+",
        type=validate_opponent,
        default=["first", "random", "greedy", "self"],
        help="first, random, greedy, self, main (unmodified main checkout), field (ladder panel) "
        "or deck:<name>",
    )
    parser.add_argument("--deck", choices=sorted(DECKS), help="Candidate deck (default deck.csv)")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="first game seed; use it to extend a run with fresh games",
    )
    parser.add_argument("--search", action="store_true", help="Baseline seat uses rollout search")
    parser.add_argument(
        "--opponent-search", action="store_true", help="self/deck opponents use rollout search"
    )
    parser.add_argument("--main-ref", type=Path, default=MAIN_REF, help="Checkout of main")
    args = parser.parse_args()
    if args.games < 1 or args.workers < 1:
        parser.error("games and workers must be positive")
    if "main" in args.opponents:
        ensure_reference(args.main_ref)
    jobs: list[tuple[str, int, int]] = []
    for name in args.opponents:
        start = args.seed_offset + len(jobs)
        if name == "field":
            jobs.extend(field_jobs(args.games, start))
        else:
            jobs.extend((name, i % 2, start + i) for i in range(args.games))
    play = partial(
        play_game,
        search=args.search,
        deck=args.deck,
        opponent_search=args.opponent_search,
        main_ref=args.main_ref,
    )
    started = time.perf_counter()
    if args.workers == 1:
        results = [play(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(play, jobs))
    wall = time.perf_counter() - started
    report = {
        "engine": SOURCE,
        "deck": args.deck or "deck.csv",
        "search": args.search,
        "opponent_search": args.opponent_search,
        "wall_seconds": wall,
        "games_per_second": len(results) / wall,
        "native_rng": "No native seed exposed; Python seed controls only random opponent",
        "summary": summary(results, wall),
        "games": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
