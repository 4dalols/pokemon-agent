"""Load the unmodified `main` agent from a git worktree as a benchmark opponent.

The agent modules share names with the working copy (`main`, `policy`, `search`, ...),
so they are imported with the worktree first on `sys.path` while the working copy's
entries are set aside, then parked in `sys.modules` under a `baseline_` prefix. Both
agents then coexist in one process, which the process-global native battle requires.
"""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from schema import Observation

ROOT = Path(__file__).resolve().parent
MODULES = ("main", "policy", "schema", "search", "value", "assets", "engine")
GENERATED = ("cards.json", "deck.csv")
DEFAULT_WORKTREE = ROOT / "results/main-worktree"


class Chooser(Protocol):
    def choose(self, observation: Observation, remaining: float | None = None) -> list[int]: ...


def ensure_worktree(path: Path, ref: str = "main") -> Path:
    """Check out `ref` into `path` (if missing) and copy the generated assets there."""
    if not (path / "main.py").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "worktree", "add", "--detach", str(path), ref], cwd=ROOT, check=True)
    for name in GENERATED:
        if not (path / name).exists():
            shutil.copy(ROOT / name, path / name)
    return path


def load_baseline(path: Path) -> Chooser:
    """Import `path/main.py` and return its `SEARCHER` without disturbing our modules."""
    ours = {name: sys.modules.pop(name) for name in MODULES if name in sys.modules}
    sys.path.insert(0, str(path))
    try:
        module = importlib.import_module("main")
    finally:
        sys.path.remove(str(path))
        for name in MODULES:
            loaded = sys.modules.pop(name, None)
            if loaded is not None:
                sys.modules[f"baseline_{name}"] = loaded
        sys.modules.update(ours)
    searcher: Chooser = module.SEARCHER
    return searcher
