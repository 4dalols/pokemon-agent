# Pokémon TCG CPU baseline

A strategy-based agent for the Pokémon TCG AI Battle Challenge (Kaggle Playground).
It plays the Mega Abomasnow ex / Kyogre deck shipped as the `kaggle-environments`
sample deck and is validated against the official Playground SDK (R2 card pool,
1431 cards) in both single games and the best-of-three Kaggle environment.

## Official files

The competition data zip (SDK source, sample submission with the `cg` Python package
and native libraries, R2 card CSVs) is licensed for competition use only and must stay
out of this public repository. Unpack it to `official-data/` (git-ignored):

```text
official-data/
  EN_Card_Data_R2_full.csv
  sample_submission/sample_submission/sample_submission/{main.py,deck.csv,cg/}
  ptcg_engine_playground/...
```

`engine.py` picks the simulator for local tooling: `$PTCG_SDK` (a directory holding
`cg/`), then the sample submission above, then the `cg` package bundled with
`kaggle-environments`. The official SDK and current `kaggle-environments` master
expose identical `AllCard`/`AllAttack` metadata.

Submission contract, taken from the official sample: `main.py` at the archive root
exposing `agent(observation: dict) -> list[int]`; the first call has
`observation["select"] is None` and must return the 60 card IDs; `deck.csv` holds one
card ID per line with no header; on Kaggle the files live in `/kaggle_simulations/agent/`.

## Reproduce the local setup

Verified on Ubuntu, Python 3.12, CPU only:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-deps 'kaggle-environments @ git+https://github.com/Kaggle/kaggle-environments@master'
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python prepare_assets.py
.venv/bin/ruff check .
.venv/bin/mypy main.py policy.py schema.py assets.py engine.py search.py prepare_assets.py benchmark.py package.py decks.py reference.py field.py tests
.venv/bin/pytest -q
.venv/bin/python benchmark.py --games 100 --workers 4 --output results/benchmark.json
.venv/bin/python package.py
```

`kaggle-environments` master is required for the best-of-three `cabt` environment and
the R2 card pool; the PyPI 1.32.7 release is a single-game build with 1267 cards.
Only the cabt engine and its needed Python dependencies are installed. An import
warning about missing `pyspiel` belongs to another environment and does not prevent
cabt battles.

`prepare_assets.py` exports metadata from the resolved native engine and writes the
deck. Candidate deck lists live in `decks.py`; `--deck <name>` selects one and
`DEFAULT_DECK` is what the submission plays. It overwrites `cards.json` and `deck.csv`. The generated files are excluded
from Git by default. The runtime agent itself uses only Python's standard library.
At runtime `PTCG_DECK_FILE` names an alternative deck file (relative to the agent
directory or absolute) to play instead of `deck.csv`.

## Strategy

The heuristic reads the deck list and card metadata, so it plays any legal deck:

- Lead with the Basic whose evolution line (within the deck) has the best attacks,
  develop evolutions, and fund the attacker whose line is closest to attacking with the
  deck's main Energy type.
- Score attacks by damage after Weakness, knockout probability and text effects
  (self-damage, Energy discards, prize-count scaling such as Supreme Overlord,
  deck-density attacks such as Hammer-lanche); prefer a guaranteed final prize.
- Score Trainers from their text: search Items by what they can fetch, draw Supporters
  by hand size and whether the hand still holds anything playable, Tools by the
  attacker they would complete, gust/switch cards by the knockout they enable.
- Preserve evolution/search pieces when paying discard costs and switch to a better
  prepared attacker when legal.
- Interpret follow-up choices as option **positions**, including iterative energy
  payments, optional searches, facedown prizes, and replacement active Pokémon.

The policy is stateless apart from immutable deck and metadata, so it carries nothing
between the games of a best-of-three match.

The heuristic does not model every opposing Ability, optimize all Trainer
combinations, or include a learned model.

## Rollout search

`search.py` wraps the heuristic with determinized rollouts on the native engine's
search API (`SearchBegin`/`SearchStep`). At each main-phase decision after turn 1 it
samples the hidden cards (own deck order and prizes, the opponent's deck, prizes and
hand, mirroring our deck list when the cards seen so far allow it), plays the top
heuristic candidates through the simulator (our heuristic acting for us, a `Policy`
built for the opponent's observed deck acting for them) until the opponent's next turn
ends, and picks the candidate with the best averaged prize/board outcome; terminal wins
and losses dominate. Every other prompt stays heuristic.

The engine is taken from `kaggle_environments.envs.cabt.cg.sim` (present in the
Kaggle runtime), falling back to a `cg` package on `sys.path`; without either the
agent is purely heuristic. The per-decision budget (`PTCG_SEARCH_BUDGET`, default
1.5 s; `PTCG_SEARCH_CANDIDATES`, default 6) shrinks with `remainingOverageTime` so a
best-of-three match stays inside the 600 s overage allowance, and the heuristic
answer is used whenever the budget is spent or the engine rejects a prediction.
`benchmark.py --search` plays the search agent against the heuristic opponents; the
test suite runs with a 0.1 s budget (`tests/conftest.py`).

## Benchmarks and replay inspection

`benchmark.py` validates selection counts and indices before every native action.
Native rejections and games exceeding 10,000 decisions raise errors rather than
being hidden as losses. Seats alternate. The default opponents use the same deck:

- `first`: take the first legal options.
- `random`: choose random legal option positions.
- `greedy`: evolve, attach (active first), play cards, then use the attack with the
  highest printed base damage. It avoids voluntary retreat and Abilities.
- `self`: the same baseline on both sides; its assigned-seat win rate is a symmetry
  check, not a measure of strength against a different agent.
- `main`: the unmodified `main` branch agent (deck, policy and search), run from a git
  worktree at `results/main-ref` (created on demand) in its own interpreter by
  `reference.py`; `--opponent-search` gives it its full search budget.
- `deck:<name>`: the current code playing another candidate from `decks.py`.
- `field`: the ladder panel in `field.py` (the most common 60-card list of each
  archetype seen in public Playground replays, with its number of entries). Games are
  allocated to archetypes in proportion to entries, each piloted by the same policy
  with the same search setting as our seat. The summary adds a `field` entry with the
  pooled W-L, Wilson 95% interval and the entry-weighted win rate next to the
  per-archetype `field:<slug>` groups. `field:<slug>` is also accepted directly.

`--deck <name>` makes the benchmarked agent play a candidate deck instead of the one in
`deck.csv`, so candidates can be round-robined against each other and against `main`.

Parallel games run in separate processes because the native battle is process-global.
The native API exposes no seed parameter. Python seeds control only the random
opponent; shuffled games and coin flips cannot be exactly replayed from those seeds.

For one game's visible decision trace:

```sh
.venv/bin/python -c 'from pathlib import Path; from benchmark import play_game; print(play_game(("greedy", 0, 42), Path("results/trace.json")))'
```

The trace contains only observations delivered to the acting agent and its chosen
indices. It is for inspection, not a deterministic simulator replay.

## Packaging

`package.py` creates `dist/submission.tar.gz`, with `main.py` and `deck.csv` at its
root plus the local modules and generated metadata. The runtime uses only the standard
library, so no `cg` package, native library, or virtual environment is embedded.
Before uploading, extract the archive elsewhere and run it through
`kaggle_environments.make("cabt", configuration={"bo": 3})` from a different working
directory in a fresh process.

## Sources

- [Competition](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/overview)
- [Official data access](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/data)
- [Simulator API](https://matsuoinstitute.github.io/cabt/api.html)
- [Current upstream implementation](https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/cabt)
