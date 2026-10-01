import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import cast

from archetypes import LIBRARY
from assets import ROOT, load_deck, load_panel
from baseline import DEFAULT_WORKTREE, Chooser, ensure_worktree, load_baseline
from decks import DECKS, deck_list
from engine import SOURCE, Battle, battle_finish, battle_select, battle_start
from main import ATTACKS, CARDS, POLICY, SEARCHER, make_searcher
from policy import Policy
from schema import Observation
from search import Searcher

OVERAGE = 600.0
SIMPLE_OPPONENTS = ("first", "random", "greedy", "self", "main", "field")
Job = tuple[str, int, int] | tuple[str, int, int, str | None]

BASELINE: dict[Path, Chooser] = {}
AGENTS: dict[tuple[int, ...], tuple[Policy, Searcher]] = {}


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
    overage_used: list[float]
    contexts: list[int]
    deck: str | None = None


def opponent_action(
    observation: Observation, kind: str, rng: random.Random, policy: Policy = POLICY
) -> list[int]:
    selection = observation["select"]
    if selection is None:
        return policy.deck.copy()
    if kind == "self":
        return policy.choose(observation)
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


def baseline(path: Path) -> Chooser:
    """The unmodified `main` agent (full search budget), loaded once per process."""
    if path not in BASELINE:
        BASELINE[path] = load_baseline(path)
    return BASELINE[path]


def agent_for(deck: list[int] | None) -> tuple[Policy, Searcher]:
    """The current policy and search for a 60-card list (`None` or our list: the shipped pair)."""
    if deck is None or sorted(deck) == sorted(POLICY.deck):
        return POLICY, SEARCHER
    key = tuple(deck)
    if key not in AGENTS:
        policy = Policy(deck, CARDS, ATTACKS)
        AGENTS[key] = policy, make_searcher(policy)
    return AGENTS[key]


def validate_opponent(name: str) -> str:
    if name in SIMPLE_OPPONENTS or (name.startswith("deck:") and name[5:] in DECKS):
        return name
    raise argparse.ArgumentTypeError(
        f"Unknown opponent {name!r}; use {', '.join(SIMPLE_OPPONENTS)} or deck:<name>"
    )


def field_schedule(weights: dict[str, int], games: int) -> list[tuple[str, int]]:
    """(deck, seat) for `games` field games split by weight (largest remainder), seats alternate."""
    total = sum(weights.values())
    shares = {name: games * weight / total for name, weight in weights.items()}
    counts = {name: int(share) for name, share in shares.items()}
    for name in sorted(shares, key=lambda name: shares[name] - counts[name], reverse=True)[
        : games - sum(counts.values())
    ]:
        counts[name] += 1
    schedule = [
        ((k + 0.5) / count, name, k % 2) for name, count in counts.items() for k in range(count)
    ]
    return [(name, seat) for _, name, seat in sorted(schedule)]


def play_game(
    job: Job,
    trace_path: Path | None = None,
    search: bool = False,
    worktree: Path = DEFAULT_WORKTREE,
    opponent_deck: list[int] | None = None,
    agent: str = "candidate",
    deck: list[int] | None = None,
    opponent_search: bool = False,
    lists: dict[str, list[int]] | None = None,
) -> GameResult:
    opponent, seat, seed = job[:3]
    deck_name = job[3] if len(job) == 4 else None
    rng = random.Random(seed)
    policy, searcher = agent_for(deck)
    chooser: Chooser = searcher if agent == "candidate" else baseline(worktree)
    started = time.perf_counter()
    decks = [policy.deck, policy.deck]
    if agent == "main":
        decks[seat] = load_deck(worktree, CARDS)
    pilot = policy
    rival: tuple[Policy, Searcher] | None = None
    if deck_name is not None:
        rival = agent_for((lists or {})[deck_name])
        pilot = rival[0]
        decks[1 - seat] = pilot.deck
    elif opponent == "main":
        decks[1 - seat] = load_deck(worktree, CARDS)
    elif opponent_deck is not None:
        decks[1 - seat] = opponent_deck
        pilot = Policy(opponent_deck, CARDS, ATTACKS)
    observation, start = battle_start(decks[0], decks[1])
    if start.errorPlayer >= 0:
        raise ValueError(f"Native deck error: {start.errorPlayer}/{start.errorType}")
    contexts: set[int] = set()
    maximum = 0.0
    opponent_maximum = 0.0
    used = [0.0, 0.0]
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
                    used,
                    sorted(contexts),
                    deck_name,
                )
            if selection is None:
                raise ValueError("Missing live selection")
            contexts.add(selection["context"])
            mover = current["yourIndex"]
            obs["remainingOverageTime"] = OVERAGE - used[mover]
            decision_started = time.perf_counter()
            if mover == seat and search:
                action = chooser.choose(obs, obs["remainingOverageTime"])
            elif mover == seat:
                action = policy.choose(obs)
            elif opponent == "main":
                action = baseline(worktree).choose(obs, obs["remainingOverageTime"])
            elif rival is not None:
                action = (
                    rival[1].choose(obs, obs["remainingOverageTime"])
                    if opponent_search
                    else rival[0].choose(obs)
                )
            else:
                action = opponent_action(obs, opponent, rng, pilot)
            duration = time.perf_counter() - decision_started
            used[mover] += duration
            if mover == seat:
                maximum = max(maximum, duration * 1000)
            else:
                opponent_maximum = max(opponent_maximum, duration * 1000)
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
    proportion = wins / n
    z = 1.96
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    margin = z * (proportion * (1 - proportion) / n + z * z / (4 * n * n)) ** 0.5 / denominator
    return [center - margin, center + margin]


def tally(group: list[GameResult]) -> dict[str, object]:
    wins = sum(result.winner == result.baseline_seat for result in group)
    draws = sum(result.winner not in (0, 1) for result in group)
    n = len(group)
    return {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": n - wins - draws,
        "win_rate": wins / n,
        "wilson_95": wilson(wins, n),
    }


def summary(results: list[GameResult]) -> dict[str, object]:
    groups: dict[str, object] = {}
    for name in sorted({result.opponent for result in results}):
        group = [result for result in results if result.opponent == name]
        n = len(group)
        decks = sorted({result.deck for result in group if result.deck is not None})
        groups[name] = {
            **tally(group),
            "decks": {
                deck: tally([result for result in group if result.deck == deck]) for deck in decks
            },
            "max_decision_ms": max(result.max_decision_ms for result in group),
            "opponent_max_decision_ms": max(result.opponent_max_decision_ms for result in group),
            "mean_game_seconds": sum(result.seconds for result in group) / n,
            "max_overage_used": max(result.overage_used[result.baseline_seat] for result in group),
            "opponent_max_overage_used": max(
                result.overage_used[1 - result.baseline_seat] for result in group
            ),
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
        type=validate_opponent,
        default=["first", "random", "greedy", "self"],
        help=f"{', '.join(SIMPLE_OPPONENTS)} or deck:<name> (a decks.py candidate)",
    )
    parser.add_argument(
        "--deck", choices=sorted(DECKS), help="decks.py candidate the measured seat plays"
    )
    parser.add_argument(
        "--opponent-search",
        action="store_true",
        help="field/deck:<name> opponents use rollout search (default: plain policy)",
    )
    parser.add_argument(
        "--main-worktree",
        type=Path,
        default=DEFAULT_WORKTREE,
        help="Checkout of the unmodified main branch used by the 'main' opponent",
    )
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--search", action="store_true", help="Baseline seat uses rollout search")
    parser.add_argument(
        "--opponent-deck",
        help="Archetype (archetypes.LIBRARY name or --opponent-decks panel name) the "
        "first/greedy/random/self opponent plays instead of our deck",
    )
    parser.add_argument(
        "--opponent-decks",
        type=Path,
        default=ROOT / "opponent_panel.json",
        help="Panel JSON played by the 'field' opponent (weights = entries) and whose deck "
        "names --opponent-deck may also use",
    )
    parser.add_argument(
        "--main-ref", default="main", help="Git ref checked out into a missing --main-worktree"
    )
    parser.add_argument(
        "--agent",
        choices=["candidate", "main"],
        default="candidate",
        help="Which search agent sits in the measured seat (requires --search)",
    )
    args = parser.parse_args()
    if args.opponent_deck is not None and "main" in args.opponents:
        parser.error("the main opponent only plays its own deck")
    if args.games < 1 or args.workers < 1:
        parser.error("games and workers must be positive")
    if "main" in args.opponents or args.agent == "main":
        ensure_worktree(args.main_worktree, args.main_ref)
    panel = load_panel(args.opponent_decks, CARDS) if args.opponent_decks.exists() else {}
    if "field" in args.opponents and not panel:
        parser.error(f"the field opponent needs the panel {args.opponent_decks}")
    lists = {name: deck for name, (deck, _) in panel.items()}
    lists.update((name, deck_list(name)) for name in DECKS)
    jobs: list[Job] = []
    for name in args.opponents:
        if name == "field":
            weights = {name: entries for name, (_, entries) in panel.items()}
            jobs.extend(
                ("field", seat, i, deck)
                for i, (deck, seat) in enumerate(field_schedule(weights, args.games))
            )
        else:
            jobs.extend(
                (name, i % 2, i, name[5:] if name.startswith("deck:") else None)
                for i in range(args.games)
            )
    opponent_deck = None
    if args.opponent_deck is not None:
        library = {entry.name: sorted(entry.cards.elements()) for entry in LIBRARY}
        library.update(lists)
        if args.opponent_deck not in library:
            parser.error(f"unknown opponent deck {args.opponent_deck!r}")
        opponent_deck = library[args.opponent_deck]
    play = partial(
        play_game,
        search=args.search,
        worktree=args.main_worktree,
        opponent_deck=opponent_deck,
        agent=args.agent,
        deck=deck_list(args.deck) if args.deck else None,
        opponent_search=args.opponent_search,
        lists=lists,
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
        "search": args.search,
        "workers": args.workers,
        "wall_seconds": wall,
        "games_per_second": len(jobs) / wall,
        "agent": args.agent,
        "deck": args.deck,
        "opponent_deck": args.opponent_deck,
        "opponent_search": args.opponent_search,
        "native_rng": "No native seed exposed; Python seed controls only random opponent",
        "summary": summary(results),
        "games": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"{len(jobs)} games in {wall:.0f} s ({len(jobs) / wall:.3f} games/s)")


if __name__ == "__main__":
    main()
