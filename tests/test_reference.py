from pathlib import Path

from reference import ReferenceAgent
from schema import Selection

ROOT = Path(__file__).resolve().parent.parent


def test_reference_agent_round_trips_deck_and_selection() -> None:
    with ReferenceAgent(ROOT) as agent:
        deck = agent.deck()
        assert len(deck) == 60
        prompt: Selection = {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [{"type": 14}],
            "deck": None,
            "contextCard": None,
            "effect": None,
        }
        assert agent.choose({"select": prompt, "current": None}) == [0]


def test_reference_agent_requires_main() -> None:
    try:
        ReferenceAgent(ROOT / "tests")
    except FileNotFoundError:
        return
    raise AssertionError("missing main.py should be rejected")


def test_reference_agent_can_be_restricted_to_heuristics() -> None:
    with ReferenceAgent(ROOT, search=False) as agent:
        assert agent.env["PTCG_SEARCH_BUDGET"] == "0"
        assert len(agent.deck()) == 60
