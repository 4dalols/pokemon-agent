import os

from assets import ROOT, load_catalog, load_deck
from imitation import BCPolicy, load_model
from policy import Policy
from schema import Observation
from search import Searcher, load_engine

CARDS, ATTACKS = load_catalog(ROOT)
DECK = load_deck(ROOT, CARDS, os.environ.get("PTCG_DECK_FILE", "deck.csv"))
MODEL = None if os.environ.get("PTCG_HEURISTIC") else load_model(ROOT)
BC_TYPES = os.environ.get("PTCG_BC_TYPES")
POLICY = (
    Policy(DECK, CARDS, ATTACKS)
    if MODEL is None
    else BCPolicy(
        DECK,
        CARDS,
        ATTACKS,
        MODEL,
        frozenset(int(kind) for kind in BC_TYPES.split(",")) if BC_TYPES else None,
    )
)
SEARCHER = Searcher(
    POLICY,
    load_engine(),
    budget=float(os.environ.get("PTCG_SEARCH_BUDGET", "1.5")),
    candidates=int(os.environ.get("PTCG_SEARCH_CANDIDATES", "6")),
)


def agent(observation: Observation) -> list[int]:
    return SEARCHER.choose(observation, observation.get("remainingOverageTime"))
