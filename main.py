import os

from assets import ROOT, load_catalog, load_deck, locate
from policy import Policy
from schema import Observation
from search import Searcher, load_engine
from value import load_model

CARDS, ATTACKS = load_catalog(ROOT)
POLICY = Policy(load_deck(ROOT, CARDS), CARDS, ATTACKS)
MODEL = (
    None
    if os.environ.get("PTCG_VALUE_MODEL", "1") == "0"
    else load_model(locate(ROOT, "value.json"))
)
SEARCHER = Searcher(
    POLICY,
    load_engine(),
    budget=float(os.environ.get("PTCG_SEARCH_BUDGET", "1.5")),
    candidates=int(os.environ.get("PTCG_SEARCH_CANDIDATES", "6")),
    model=MODEL,
)


def agent(observation: Observation) -> list[int]:
    return SEARCHER.choose(observation, observation.get("remainingOverageTime"))
