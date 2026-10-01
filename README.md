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
.venv/bin/python prepare_assets.py --deck dragapult
.venv/bin/ruff check .
.venv/bin/mypy main.py policy.py schema.py assets.py engine.py search.py prepare_assets.py benchmark.py package.py imitation.py train_bc.py tests
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
deck. `--deck dragapult` (default; the list played by the top-rated Playground teams)
or `--deck abomasnow` (the earlier sample list) selects it. It overwrites `cards.json`
and `deck.csv` and also keeps a `deck-<name>.csv` copy, which `PTCG_DECK_FILE` can
point `main.py` at for benchmarks. The generated files are excluded
from Git by default. The runtime agent itself uses only Python's standard library.

## Strategy

- Lead with Snover when available, develop its evolution, and fund the active attacker.
- Use Mega Abomasnow ex's Hammer-lanche with an estimated remaining Water Energy
  density and knockout probability; prefer Frost Barrier when it scores better.
- Use Kyogre's Riptide after Water Energy accumulates in the discard pile.
- Preserve evolution/search pieces when paying discard costs, draw out of small hands,
  attach Powerglass, and switch to a better prepared attacker when legal.
- Interpret follow-up choices as option **positions**, including iterative energy
  payments, optional searches, facedown prizes, and replacement active Pokémon.

The policy is stateless apart from immutable deck and metadata, so it carries nothing
between the games of a best-of-three match.

The heuristic does not model every opposing Ability, optimize all Trainer
combinations, or include a learned model.

## Imitation policy

`imitation.py` ranks options with a linear model over sparse string features learned
from the public Playground replays (`train_bc.py`, offline): the selection kind and
context, the option (card identity, type, stage, HP and damage buckets, attached
energy, attack identity, printed damage and whether it knocks out the defender,
target zone and owner), the visible board (turn, hand and bench sizes, prizes, deck
size, both actives, bench and hand contents) and the selections already made this
turn. For optional prompts a virtual "decline" row is scored alongside the options,
so `Policy.choose`'s "take options scoring above zero" rule keeps working.
`BCPolicy` reuses `Policy.choose` (positions, iterative energy payments, counts) and
only replaces `scores()`; inference is pure Python (about 0.4 ms per decision) from
`bc_model.json`, which ships in the archive. `main.py` falls back to the heuristic
policy when the model file is missing or `PTCG_HEURISTIC` is set.

Training data: every decision of agents whose episode `avg_score` is at least 800
(or who belong to the top teams), with the Dragapult list weighted twice; 20% of the
episodes are held out by hash. Held-out top-1 accuracy per selection type against the
heuristic policy on the same prompts is written to `results/bc_eval.json`.

```sh
.venv/bin/python -m pip install kaggle
KAGGLE_API_TOKEN=... .venv/bin/kaggle datasets download -d kaggle/the-pokemon-company-ptcg-ai-battle-challenge-playground-episodes-2026-09-29 --unzip -p ../ptcg-data/episodes
.venv/bin/python train_bc.py ../ptcg-data/decisions   # decisions extracted from the episode JSONs
```

## Rollout search

`search.py` wraps the heuristic with determinized rollouts on the native engine's
search API (`SearchBegin`/`SearchStep`). At each main-phase decision after turn 1 it
samples the hidden cards (own deck order and prizes, the opponent's deck, prizes and
hand, mirroring our deck list when the cards seen so far allow it), plays the top
policy candidates through the simulator with the policy acting for both sides until
the opponent's next turn ends, and picks the candidate with the best averaged
prize/board outcome; terminal wins and losses dominate. The policy is the imitation
ranker when its model is present (its same-turn history is snapshotted before the
rollouts and only the played action is committed), otherwise the heuristic. Every
other prompt is answered by the policy directly.

The engine is taken from `kaggle_environments.envs.cabt.cg.sim` (present in the
Kaggle runtime), falling back to a `cg` package on `sys.path`; without either the
agent is purely heuristic. The per-decision budget (`PTCG_SEARCH_BUDGET`, default
1.5 s; `PTCG_SEARCH_CANDIDATES`, default 6) shrinks with `remainingOverageTime` so a
best-of-three match stays inside the 600 s overage allowance, and the heuristic
answer is used whenever the budget is spent or the engine rejects a prediction.
`benchmark.py --search` plays the search agent against the heuristic opponents, and
`--opponents main --main-root ../pokemon-agent-main` against the unmodified `main`
branch running as a separate process from a git worktree (prepare its `cards.json`
and `deck.csv` there first); the test suite runs with a 0.1 s budget and the
heuristic agent (`tests/conftest.py`), the learned policy is covered by
`tests/test_imitation.py`.

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
