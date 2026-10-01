"""Load a frozen copy of this agent from a git ref for head-to-head benchmarks.

The ref is exported with ``git archive`` into a cache directory (together with the
generated ``cards.json`` and ``deck.csv``) and imported under its own module objects
so the frozen policy and searcher coexist with the working copy in one process.
"""

import importlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

from schema import Observation

MODULES = ("assets", "schema", "policy", "search", "main")
GENERATED = ("cards.json", "deck.csv")

Agent = Callable[[Observation], list[int]]


def export(ref: str, root: Path, cache: Path) -> Path:
    target = cache / ref.replace("/", "_")
    if not (target / "main.py").exists():
        archive = subprocess.run(
            ["git", "-C", str(root), "archive", ref], check=True, capture_output=True
        ).stdout
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(target, filter="data")
    for name in GENERATED:
        if not (target / name).exists():
            shutil.copy(root / name, target / name)
    return target


def load_agent(directory: Path) -> Agent:
    """Import ``main.agent`` from ``directory`` with its stock settings (no PTCG_* overrides)."""
    saved = {name: sys.modules.pop(name) for name in MODULES if name in sys.modules}
    overrides = {key: os.environ.pop(key) for key in list(os.environ) if key.startswith("PTCG_")}
    sys.path.insert(0, str(directory))
    try:
        agent = cast(Agent, importlib.import_module("main").agent)
    finally:
        sys.path.remove(str(directory))
        os.environ.update(overrides)
        for name in MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return agent
