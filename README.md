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
.venv/bin/mypy main.py policy.py schema.py assets.py engine.py search.py value.py train_value.py baseline.py memory.py archetypes.py prediction.py prepare_assets.py benchmark.py package.py tests
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
deck. It overwrites `cards.json` and `deck.csv`; keep
modified deck lists elsewhere before rerunning it. The generated files are excluded
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

## Rollout search

`search.py` wraps the heuristic with determinized rollouts on the native engine's
search API (`SearchBegin`/`SearchStep`). At every searchable prompt after turn 1
(main phase, and card prompts such as switch/promote, bench, evolution, attach and
discard targets; energy-payment prompts stay heuristic) it samples the hidden cards
(own deck order and prizes, the opponent's deck, prizes and hand, mirroring our deck
list when the cards seen so far allow it), plays distinct candidate answers through
the simulator with the heuristic acting for both sides until the opponent's next turn
ends, and picks the candidate with the best averaged leaf value; terminal wins and
losses dominate. Candidates are the heuristic answer plus its single-card variations,
and successive halving drops the weaker half of the field at fixed fractions of the
time budget so the survivors get more determinizations.

## Learned leaf evaluation

`value.py` featurizes a `Current` state from one player's point of view (prizes,
active/bench HP, damage and energy, affordable attack damage after weakness and
resistance, knockout availability, hand/deck/discard sizes, special conditions, turn
and side to move) and evaluates it with a one-hidden-layer MLP in pure Python; the
weights live in `value.json` (about 2k parameters) and are loaded at import time by
`main.py`. When present, the model's win probability replaces the hand-written
prize/board score at rollout leaves (`PTCG_VALUE_MODEL=0` restores the heuristic
score). `train_value.py` regenerates the weights reproducibly: it plays seeded
self-play games on the native engine (heuristic, short-budget search and the benchmark
opponents, both seats), labels every MAIN-selection state of both players with the
final result, and fits the network with numpy, holding out 20% of the games to report
log-loss and accuracy next to the prior and the heuristic score:

```sh
.venv/bin/python train_value.py --games 8000 --workers 8 --dataset results/dataset.json
```

The engine is taken from `kaggle_environments.envs.cabt.cg.sim` (present in the
Kaggle runtime), falling back to a `cg` package on `sys.path`; without either the
agent is purely heuristic. The per-decision budget (`PTCG_SEARCH_BUDGET`, default
1.5 s; `PTCG_SEARCH_CANDIDATES`, default 8) is capped from `remainingOverageTime`
(pool minus a 60 s reserve, spread over 40 decisions per remaining game) so a
best-of-three match stays inside the 600 s overage allowance, and the heuristic
answer is used whenever the budget is spent or the engine rejects a prediction.
Environment switches for experiments: `PTCG_SEARCH_PROMPTS=main` (main phase only),
`PTCG_SEARCH_HALVING=0`, `PTCG_SEARCH_HORIZON=2` (rollouts through two of our turns),
`PTCG_SEARCH_EPSILON=0.2` (epsilon-greedy rollout policy) and `PTCG_SEARCH_MODEL=1`
(bias the opponent's sampled hand away from cards they would have played last turn).
`benchmark.py --search` plays the search agent against the heuristic opponents; the
test suite runs with a 0.1 s budget (`tests/conftest.py`).

`memory.Tracker` keeps per-game memory from the incremental `observation["logs"]`
(reset whenever `round` changes or `turn` goes backwards): opponent card identities
by serial with per-zone probabilities, our exact deck and prizes once a full-deck
search has shown the whole deck, attacks, damage, coin flips and energy attachments.
`archetypes.Predictor` holds a library of 60-card lists (the top Playground
Dragapult ex list first, then lists reconstructed from public replays, plus a mirror
of our own deck) and weights them by how many revealed opponent cards each list
fails to explain; `search.Searcher.begin` samples the opponent's hidden deck, hand
and prizes from that mixture and uses our exact prizes when known, falling back to
the uniform pool whenever logs are missing or inconsistent. `Tracker.threat` feeds
the heuristics (bench size against bench snipers, keeping Pokémon within revealed
damage range out of the active spot, preferring attackers that hit weakness).
`prediction.py` reports the log-likelihood of the opponent's actual draws under the
old uniform predictor and the tracked mixture.

## Benchmarks and replay inspection

`benchmark.py` validates selection counts and indices before every native action.
Native rejections and games exceeding 10,000 decisions raise errors rather than
being hidden as losses. Seats alternate. Each side is handed a simulated 600 s
`remainingOverageTime` pool that shrinks with its own decision time, and the report
includes the largest per-game overage spend. The default opponents use the same deck:

- `first`: take the first legal options.
- `random`: choose random legal option positions.
- `greedy`: evolve, attach (active first), play cards, then use the attack with the
  highest printed base damage. It avoids voluntary retreat and Abilities.
- `self`: the same baseline on both sides; its assigned-seat win rate is a symmetry
  check, not a measure of strength against a different agent.
- `main`: the unmodified agent from the `main` branch with its full search budget,
  imported from a git worktree (`baseline.py`, default `results/main-worktree`,
  created on demand). Use `--search --opponents main` to compare a change head-to-head.

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
