import argparse
import importlib
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import cast

from engine import SOURCE, Battle, battle_finish, battle_select, battle_start
from main import ATTACKS, CARDS, POLICY
from policy import Policy
from prepare_assets import DECKS, DEFAULT_DECK, deck_ids
from schema import Observation
from search import Searcher, load_engine

Agent = Callable[[Observation], list[int]]


def load_main(worktree: Path) -> Agent:
    names = ("main", "assets", "policy", "search", "schema")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    sys.path.insert(0, str(worktree.resolve()))
    try:
        module = importlib.import_module("main")
        return cast(Agent, module.agent)
    finally:
        sys.path.pop(0)
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


@dataclass
class GameResult:
    opponent: str
    baseline_seat: int
    winner: int
    steps: int
    turns: int
    max_decision_ms: float
    seconds: float
    contexts: list[int]
    deck: str = "abomasnow"
    seconds_used: float = 0.0
    opponent_seconds_used: float = 0.0
    opponent_max_decision_ms: float = 0.0


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


def play_game(
    job: tuple[str, int, int],
    trace_path: Path | None = None,
    search: bool = False,
    deck: str = "abomasnow",
    main_worktree: Path | None = None,
    budget: float = 1.5,
) -> GameResult:
    opponent, seat, seed = job
    rng = random.Random(seed)
    started = time.perf_counter()
    policy = Policy(deck_ids(deck), CARDS, ATTACKS)
    searcher = Searcher(policy, load_engine(), budget=budget)
    enemy_policy = Policy(deck_ids(opponent) if opponent in DECKS else policy.deck, CARDS, ATTACKS)
    enemy_searcher = Searcher(enemy_policy, load_engine(), budget=budget)

    def search_agent(obs: Observation) -> list[int]:
        return enemy_searcher.choose(obs, obs.get("remainingOverageTime"))

    enemy_agent: Agent = search_agent
    if opponent == "main":
        if main_worktree is None:
            raise ValueError("--main-worktree is required for the frozen main opponent")
        enemy_agent = load_main(main_worktree)
        enemy_deck = enemy_agent({"select": None, "current": None})
    else:
        enemy_deck = enemy_policy.deck
    decks = [policy.deck, enemy_deck] if seat == 0 else [enemy_deck, policy.deck]
    observation, start = battle_start(*decks)
    if start.errorPlayer >= 0:
        raise ValueError(f"Native deck error: {start.errorPlayer}/{start.errorType}")
    contexts: set[int] = set()
    maximum = 0.0
    elapsed = [0.0, 0.0]
    enemy_maximum = 0.0
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
                    time.perf_counter() - started,
                    sorted(contexts),
                    deck,
                    elapsed[seat],
                    elapsed[1 - seat],
                    enemy_maximum,
                )
            if selection is None:
                raise ValueError("Missing live selection")
            contexts.add(selection["context"])
            actor = current["yourIndex"]
            obs["remainingOverageTime"] = 600 - elapsed[actor]
            decision_started = time.perf_counter()
            if actor == seat and search:
                action = searcher.choose(obs, obs["remainingOverageTime"])
            elif actor == seat or opponent == "self":
                action = policy.choose(obs)
            elif opponent in DECKS or opponent == "main":
                action = enemy_agent(obs)
            else:
                action = opponent_action(obs, opponent, rng)
            duration = time.perf_counter() - decision_started
            elapsed[actor] += duration
            if elapsed[actor] > 600:
                raise RuntimeError(f"Seat {actor} exceeded its 600 second allowance")
            if actor == seat:
                maximum = max(maximum, duration * 1000)
            else:
                enemy_maximum = max(enemy_maximum, duration * 1000)
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


def summary(results: list[GameResult]) -> dict[str, object]:
    groups: dict[str, object] = {}
    for name in sorted({result.opponent for result in results}):
        group = [result for result in results if result.opponent == name]
        wins = sum(result.winner == result.baseline_seat for result in group)
        draws = sum(result.winner not in (0, 1) for result in group)
        n = len(group)
        proportion = wins / n
        z = 1.96
        denominator = 1 + z * z / n
        center = (proportion + z * z / (2 * n)) / denominator
        margin = z * (proportion * (1 - proportion) / n + z * z / (4 * n * n)) ** 0.5 / denominator
        groups[name] = {
            "games": n,
            "wins": wins,
            "draws": draws,
            "losses": n - wins - draws,
            "win_rate": proportion,
            "wilson_95": [center - margin, center + margin],
            "max_decision_ms": max(result.max_decision_ms for result in group),
            "opponent_max_decision_ms": max(result.opponent_max_decision_ms for result in group),
            "max_seconds_used": max(result.seconds_used for result in group),
            "opponent_max_seconds_used": max(result.opponent_seconds_used for result in group),
            "mean_turns": sum(result.turns for result in group) / n,
            "seats": dict(Counter(result.baseline_seat for result in group)),
        }
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=40, help="Games per opponent")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--deck", choices=DECKS, default=DEFAULT_DECK)
    parser.add_argument("--main-worktree", type=Path)
    parser.add_argument("--round-robin", action="store_true")
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=["first", "random", "greedy", "self", "main", *DECKS],
        default=["first", "random", "greedy", "self"],
    )
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--search", action="store_true", help="Baseline seat uses rollout search")
    args = parser.parse_args()
    if args.games < 1 or args.workers < 1:
        parser.error("games and workers must be positive")
    if args.round_robin:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for left, right in combinations(DECKS, 2):
            path = args.output.with_name(f"{left}-vs-{right}.json")
            jobs = [(right, i % 2, i) for i in range(args.games)]
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                results = list(executor.map(partial(play_game, search=True, deck=left), jobs))
            report = {
                "engine": SOURCE,
                "search": True,
                "budget": 1.5,
                "deck": left,
                "native_rng": "No native seed exposed; seats alternate, games are independent",
                "summary": summary(results),
                "games": [asdict(r) for r in results],
            }
            path.write_text(json.dumps(report, indent=2))
            print(left, right, json.dumps(report["summary"]), flush=True)
        return
    jobs = [(name, i % 2, i) for name in args.opponents for i in range(args.games)]
    if args.workers == 1:
        results = [
            play_game(job, search=args.search, deck=args.deck, main_worktree=args.main_worktree)
            for job in jobs
        ]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                executor.map(
                    partial(
                        play_game,
                        search=args.search,
                        deck=args.deck,
                        main_worktree=args.main_worktree,
                    ),
                    jobs,
                )
            )
    report = {
        "engine": SOURCE,
        "search": args.search,
        "budget": 1.5,
        "deck": args.deck,
        "native_rng": "No native seed exposed; Python seed controls only random opponent",
        "summary": summary(results),
        "games": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
