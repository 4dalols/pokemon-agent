# Pokémon TCG CPU baseline

This is a local, strategy-based agent for the **public `kaggle-environments==1.32.7`
cabt simulator**. It plays the public sample Mega Abomasnow ex / Kyogre deck.
The archive is a **candidate for adaptation**, not a verified Playground submission.

## Current competition gate

The official Playground SDK, sample submission, and English R2 card files are still
needed. They must establish the actual entrypoint, deck CSV format, eligible cards,
best-of-three transitions, and runtime limits. No Kaggle upload has been attempted.
The pinned public release runs single games; current GitHub master has a different
catalog and best-of-three interpreter. Do not silently mix their metadata.

`deck.csv` currently uses `card_id,count` for this project's local tooling. Its
schema has not been compared with the official sample. The Python entrypoint is
`agent(observation) -> list[int]`; initialization returns the 60 card IDs.

## Reproduce the local setup

Verified on Ubuntu, Python 3.12, CPU only:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-deps kaggle-environments==1.32.7
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python prepare_assets.py
.venv/bin/ruff check .
.venv/bin/mypy main.py policy.py schema.py assets.py prepare_assets.py benchmark.py package.py tests
.venv/bin/pytest -q
.venv/bin/python benchmark.py --games 60 --workers 2 --output results/benchmark.json
.venv/bin/python package.py
```

Only the cabt engine and its needed Python dependencies are installed. An import
warning about missing `pyspiel` belongs to another environment and does not prevent
cabt battles. Installing every Kaggle environment's optional dependencies is not
required for this project.

`prepare_assets.py` exports metadata from the exact installed native engine and
generates the public sample deck. It overwrites `cards.json` and `deck.csv`; keep
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

The policy is stateless apart from immutable deck and metadata, so it does not reuse
turn counters or serial-number caches between games. This does not establish that
the official best-of-three wrapper is compatible.

This is a starting heuristic. It does not model every opposing Ability, optimize
all Trainer combinations, perform simulator search, or include a learned model.

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

`package.py` creates `dist/public-baseline.tar.gz`, with `main.py` and `deck.csv`
at its root plus its local modules and generated metadata. No virtual environment,
native library, credentials, or training data is embedded. Local checks should also
run the extracted archive through Kaggle's file-based agent loader in a fresh process.

Keep restricted competition files out of a public repository. The source project
can be moved into a repository once a destination is chosen; no unrelated repository
has been changed.

## Sources

- [Competition](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/overview)
- [Official data access](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/data)
- [Simulator API](https://matsuoinstitute.github.io/cabt/api.html)
- [Pinned public package](https://pypi.org/project/kaggle-environments/1.32.7/)
- [Current upstream implementation](https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/cabt)
