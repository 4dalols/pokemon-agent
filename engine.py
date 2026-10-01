"""Locate a cabt simulator for local testing.

Resolution order: the `PTCG_SDK` environment variable (a directory containing the
official `cg/` package), the official sample submission under `official-data/`, then
the `cg` package bundled with `kaggle_environments`. Both expose the same module API.
"""

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent
OFFICIAL_SDK = ROOT / "official-data/sample_submission/sample_submission/sample_submission"


def _load() -> tuple[ModuleType, ModuleType, str]:
    candidate = os.environ.get("PTCG_SDK")
    sdk = Path(candidate) if candidate else OFFICIAL_SDK
    if (sdk / "cg" / "sim.py").exists():
        sys.path.append(str(sdk))
        game, sim = importlib.import_module("cg.game"), importlib.import_module("cg.sim")
        return game, sim, f"official SDK at {sdk}"
    return (
        importlib.import_module("kaggle_environments.envs.cabt.cg.game"),
        importlib.import_module("kaggle_environments.envs.cabt.cg.sim"),
        "kaggle_environments bundled engine",
    )


_game, _sim, SOURCE = _load()
battle_start = _game.battle_start
battle_select = _game.battle_select
battle_finish = _game.battle_finish
Battle = _sim.Battle
lib = _sim.lib
