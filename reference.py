"""Run an unmodified agent checkout (e.g. a git worktree of main) in a subprocess."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

from schema import Observation

SERVER = """
import json, os, sys
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])
replies = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
from main import agent
for line in sys.stdin:
    replies.write(json.dumps(agent(json.loads(line))) + "\\n")
    replies.flush()
"""


class ReferenceAgent:
    """Drives ``main.agent`` from another checkout over a JSON-lines pipe.

    The checkout is imported in its own interpreter with its own working directory, so its
    modules, catalog, deck and native search state never collide with the current code.
    """

    def __init__(self, root: Path, python: str = sys.executable, search: bool = True) -> None:
        self.root = root.resolve()
        if not (self.root / "main.py").exists():
            raise FileNotFoundError(f"No main.py in reference checkout {self.root}")
        self.env = dict(os.environ)
        if not search:
            self.env["PTCG_SEARCH_BUDGET"] = "0"
        self.process = subprocess.Popen(
            [python, "-c", SERVER, str(self.root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=self.env,
        )

    def choose(self, observation: Observation) -> list[int]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Reference agent pipes are closed")
        self.process.stdin.write(json.dumps(observation) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Reference agent at {self.root} exited: {self.process.poll()}")
        return cast(list[int], json.loads(line))

    def deck(self) -> list[int]:
        return self.choose({"select": None, "current": None})

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=30)

    def __enter__(self) -> "ReferenceAgent":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def ensure_reference(root: Path, python: str = sys.executable, ref: str = "main") -> Path:
    """Create a git worktree of ``ref`` at ``root`` (with generated assets) if missing."""
    if not (root / "main.py").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "worktree", "add", "--detach", str(root), ref], check=True)
    if not (root / "deck.csv").exists() or not (root / "cards.json").exists():
        subprocess.run([python, "prepare_assets.py"], cwd=root, check=True)
    return root
