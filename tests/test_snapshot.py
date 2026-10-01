from pathlib import Path

from main import POLICY
from snapshot import export, load_agent

ROOT = Path(__file__).resolve().parent.parent


def test_frozen_agent_loads_from_a_git_ref(tmp_path: Path) -> None:
    directory = export("HEAD", ROOT, tmp_path)
    assert (directory / "main.py").exists() and (directory / "deck.csv").exists()
    agent = load_agent(directory)
    assert agent({"select": None, "current": None}) == POLICY.deck
    import main

    assert main.POLICY is POLICY
