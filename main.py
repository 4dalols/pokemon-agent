from assets import ROOT, load_catalog, load_deck
from policy import Policy
from schema import Observation

CARDS, ATTACKS = load_catalog(ROOT)
POLICY = Policy(load_deck(ROOT, CARDS), CARDS, ATTACKS)


def agent(observation: Observation) -> list[int]:
    return POLICY.choose(observation)
