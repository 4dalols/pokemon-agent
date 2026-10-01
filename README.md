# Pokémon TCG CPU baseline

A strategy-based agent for the Pokémon TCG AI Battle Challenge (Kaggle Playground).
The default remains Mega Abomasnow/Kyogre after the four-deck round robin. The exact
Playground Dragapult and Hydrapple/Ogerpon/Meganium lists and a Mega Kangaskhan/Slowking
port remain selectable for comparison. None of the tested finalists beat frozen main;
these changes are an experimental draft. Validation uses the official Playground SDK
(R2 card pool, 1431 cards) and the best-of-three Kaggle environment.

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
.venv/bin/mypy main.py policy.py schema.py assets.py engine.py search.py prepare_assets.py benchmark.py package.py tests
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

The Dragapult policy evolves Dreepy through Drakloak, uses Recon Directive to
develop its board, funds Phantom Dive with Fire and Psychic Energy, and allocates
its six counters across reachable knockouts. It attaches Darkness Energy to
Munkidori for Adrena-Brain, opens with Budew for Item lock, uses Crushing Hammer
on low-Energy attackers, and prioritizes Boss targets that yield Prizes.
Xerosic's value grows with the opponent's hand size. These are heuristic values,
not a complete matchup model or exact multi-turn solver.

The original Abomasnow strategy is retained:

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
search API (`SearchBegin`/`SearchStep`). At each main-phase decision after turn 1 it
samples the hidden cards (own deck order and prizes, the opponent's deck, prizes and
hand, mirroring our deck list when the cards seen so far allow it), plays the top
heuristic candidates through the simulator with the heuristic acting for both sides
until the opponent's next turn ends, and picks the candidate with the best averaged
prize/board outcome; terminal wins and losses dominate. Every other prompt stays
heuristic.

The engine is taken from `kaggle_environments.envs.cabt.cg.sim` (present in the
Kaggle runtime), falling back to a `cg` package on `sys.path`; without either the
agent is purely heuristic. The per-decision budget (`PTCG_SEARCH_BUDGET`, default
1.5 s; `PTCG_SEARCH_CANDIDATES`, default 6) shrinks with `remainingOverageTime` so a
best-of-three match stays inside the 600 s overage allowance, and the heuristic
answer is used whenever the budget is spent or the engine rejects a prediction.
`benchmark.py --search` plays the search agent against the heuristic opponents; the
test suite runs with a 0.1 s budget (`tests/conftest.py`).

## Benchmarks and replay inspection

### Ported archetypes

`prepare_assets.DECKS` contains `abomasnow`, `dragapult`, `kangaskhan`, and
`hydrapple`. Generate a particular submission deck with
`PTCG_DECK=hydrapple .venv/bin/python prepare_assets.py` before packaging.
The environment variable selects the asset generation step; the deployed agent
reads the generated `deck.csv`. No SDK files are included in the archive.

The ports preserve each archetype's core rather than copying tournament lists
verbatim. Dragapult uses Recon Directive and Phantom Dive; Kangaskhan/Slowking
uses Run Errand, Academy at Night and Ciphermaniac's Codebreaking to copy Kyurem
or Zeraora; Hydrapple/Ogerpon uses Grass acceleration and board-wide damage scaling.
The existing main-phase search is unchanged. Follow-up decisions use the policy,
including colored Energy requirements and damage counters that prioritize
reachable knockouts without targeting already defeated Pokémon.

Sources and adaptation notes:

- [Previous-round meta report](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/737107)
- [Dragapult R2 IDs](https://raw.githubusercontent.com/frankiesardo/extreme-speed/main/decks/dragapult.txt)
  and [tournament reference](https://www.limitlesstcg.com/decks/list/27265): the
  final `dragapult` candidate is the exact 60-card Playground top-two list
  supplied on 2026-10-01. Every supplied ID and name maps to R2. It retains
  Munkidori/Darkness, Budew, four Crushing Hammer, two Xerosic's Machinations
  and two Jamming Tower. The preliminary simplified port was superseded before
  the final evaluation. Counter allocation maximizes reachable Prizes across
  multiple knockouts; Munkidori removes damage from the friendly board and
  targets opposing knockouts. Energy attachment funds Adrena-Brain separately
  from Munkidori's attack.
- [Kangaskhan/Slowking reference](https://www.limitlesstcg.com/decks/list/28251):
  **Telepathic Psychic Energy is the only unmapped name in these three source
  lists**; replace it with Basic Psychic Energy. Omit Metagross and Lillie's
  Clefairy ex, increase Kyurem/Zeraora and Energy, and simplify the Tool package.
  Those omitted Pokémon and Tools are present in R2, not mapping failures.
- [Hydrapple/Ogerpon reference](https://www.limitlesstcg.com/decks/list/27727):
  the final candidate is the exact Steve421471 public-replay list supplied on
  2026-10-01, including Meganium, Tapu Bulu and four Forest of Vitality. Every
  supplied ID and name maps to R2. Wild Growth and Forest of Vitality are passive
  effects handled by the engine; the policy reads its effective Energy counts
  and available evolution options. Alakazam was not evaluated.
- [Official simulator differences](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708586):
  legal options are authoritative (some attacks with no applicable effect are
  absent), simultaneous final Prize claims are draws, and some target ordering
  is automatic. The agent selects only offered options.

Run the full four-deck round robin (100 independent games per pairing, 50 in
each seat; both sides use default 1.5-second search budgets):

```sh
.venv/bin/python benchmark.py --round-robin --games 100 --workers 8 --search
```

To compare against a frozen, unmodified main, create a detached worktree at the
desired commit, generate its original assets inside that worktree, and pass its
absolute path. The loader imports that worktree's `main`, `assets`, `policy`,
`schema`, and `search` together, then restores the candidate's modules.

```sh
.venv/bin/python benchmark.py --deck hydrapple --search --games 100 --workers 8 \
  --opponents main first greedy random --main-worktree /path/to/frozen-main \
  --output results/hydrapple-validation.json
```

JSON reports include raw games, seats, wins/losses/draws, Wilson 95% intervals,
maximum decision latency, and maximum per-game overage use for both players.
The single-game benchmark starts each seat with 600 seconds; the separately
validated packaged best-of-three shares 600 seconds across the entire match.
Unsetting `PTCG_SEARCH_BUDGET` and `PTCG_SEARCH_CANDIDATES` preserves frozen main's
defaults. Weak opponents use the candidate deck; deck-vs-deck opponents use their
named deck. Native randomness is not seedable, so seat balance does not mean
identical shuffled draws.

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
