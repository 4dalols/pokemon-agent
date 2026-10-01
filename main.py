import os
from pathlib import Path

from assets import ROOT, load_catalog, load_deck, locate
from memory import Tracker
from policy import Policy
from schema import Observation
from search import Searcher, load_engine
from value import load_model

CARDS, ATTACKS = load_catalog(ROOT)
POLICY = Policy(load_deck(ROOT, CARDS), CARDS, ATTACKS)
VALUE_SETTING = os.environ.get("PTCG_VALUE_MODEL", "1")
MODEL = (
    None
    if VALUE_SETTING == "0"
    else load_model(Path(VALUE_SETTING) if VALUE_SETTING != "1" else locate(ROOT, "value.json"))
)


def make_searcher(policy: Policy) -> Searcher:
    """The shipped search configuration (environment overrides included) for a deck's policy."""
    return Searcher(
        policy,
        load_engine(),
        budget=float(os.environ.get("PTCG_SEARCH_BUDGET", "1.5")),
        candidates=int(os.environ.get("PTCG_SEARCH_CANDIDATES", "8")),
        prompts=os.environ.get("PTCG_SEARCH_PROMPTS", "all"),
        halving=os.environ.get("PTCG_SEARCH_HALVING", "1") == "1",
        horizon=int(os.environ.get("PTCG_SEARCH_HORIZON", "1")),
        epsilon=float(os.environ.get("PTCG_SEARCH_EPSILON", "0")),
        model=MODEL,
        tracker=Tracker(policy.deck),
    )


SEARCHER = make_searcher(POLICY)


def agent(observation: Observation) -> list[int]:
    return SEARCHER.choose(observation, observation.get("remainingOverageTime"))
