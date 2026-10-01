import os

from assets import ROOT, load_catalog, load_deck
from policy import Policy
from schema import Observation
from search import Searcher, load_engine

CARDS, ATTACKS = load_catalog(ROOT)
POLICY = Policy(load_deck(ROOT, CARDS), CARDS, ATTACKS)
SEARCHER = Searcher(
    POLICY,
    load_engine(),
    budget=float(os.environ.get("PTCG_SEARCH_BUDGET", "1.5")),
    candidates=int(os.environ.get("PTCG_SEARCH_CANDIDATES", "6")),
)


def agent(observation: Observation) -> list[int]:
    return SEARCHER.choose(observation, observation.get("remainingOverageTime"))
