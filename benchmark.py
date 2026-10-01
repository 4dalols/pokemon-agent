import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import cast

from engine import SOURCE, Battle, battle_finish, battle_select, battle_start
from main import ATTACKS, POLICY, SEARCHER
from schema import Observation


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


WORKER = """
import json, sys
out, sys.stdout = sys.stdout, sys.stderr
sys.path.insert(0, ".")
import main
for line in sys.stdin:
    out.write(json.dumps(main.agent(json.loads(line))) + "\\n")
    out.flush()
"""


class RemoteAgent:
    """A full `main.agent` (e.g. the unmodified main branch) in its own process."""

    def __init__(self, root: Path) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-c", WORKER],
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    def act(self, observation: Observation) -> list[int]:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(observation) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Remote agent exited")
        return cast(list[int], json.loads(line))

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        self.process.wait(timeout=30)


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
    main_root: Path | None = None,
) -> GameResult:
    opponent, seat, seed = job
    rng = random.Random(seed)
    started = time.perf_counter()
    remote = RemoteAgent(main_root) if opponent == "main" and main_root else None
    enemy_deck = remote.act({"select": None, "current": None}) if remote else POLICY.deck
    decks = (POLICY.deck, enemy_deck) if seat == 0 else (enemy_deck, POLICY.deck)
    observation, start = battle_start(*decks)
    if start.errorPlayer >= 0:
        raise ValueError(f"Native deck error: {start.errorPlayer}/{start.errorType}")
    contexts: set[int] = set()
    maximum = 0.0
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
                )
            if selection is None:
                raise ValueError("Missing live selection")
            contexts.add(selection["context"])
            decision_started = time.perf_counter()
            if current["yourIndex"] == seat and search:
                action = SEARCHER.choose(obs)
                maximum = max(maximum, (time.perf_counter() - decision_started) * 1000)
            elif current["yourIndex"] == seat or opponent == "self":
                action = POLICY.choose(obs)
                maximum = max(maximum, (time.perf_counter() - decision_started) * 1000)
            elif remote is not None:
                action = remote.act(obs)
            else:
                action = opponent_action(obs, opponent, rng)
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
        if remote is not None:
            remote.close()
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
            "mean_turns": sum(result.turns for result in group) / n,
            "seats": dict(Counter(result.baseline_seat for result in group)),
        }
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=40, help="Games per opponent")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=["first", "random", "greedy", "self", "main"],
        default=["first", "random", "greedy", "self"],
    )
    parser.add_argument(
        "--main-root",
        type=Path,
        default=Path("../pokemon-agent-main"),
        help="Checkout (git worktree) of the unmodified main branch for the 'main' opponent",
    )
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--search", action="store_true", help="Baseline seat uses rollout search")
    args = parser.parse_args()
    if args.games < 1 or args.workers < 1:
        parser.error("games and workers must be positive")
    if "main" in args.opponents and not (args.main_root / "main.py").exists():
        parser.error(f"no main checkout at {args.main_root}")
    jobs = [(name, i % 2, i) for name in args.opponents for i in range(args.games)]
    play = partial(play_game, search=args.search, main_root=args.main_root)
    if args.workers == 1:
        results = [play(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(play, jobs))
    report = {
        "engine": SOURCE,
        "search": args.search,
        "policy": type(POLICY).__name__,
        "deck": sorted(Counter(POLICY.deck).items()),
        "native_rng": "No native seed exposed; Python seed controls only random opponent",
        "summary": summary(results),
        "games": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
