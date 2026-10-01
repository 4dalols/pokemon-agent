import csv
import json
from collections import Counter
from pathlib import Path
from typing import cast

from schema import AttackData, CardData, Catalog

ROOT = Path(__file__).resolve().parent


def load_catalog(root: Path) -> tuple[dict[int, CardData], dict[int, AttackData]]:
    catalog = cast(Catalog, json.loads((root / "cards.json").read_text()))
    return (
        {card["cardId"]: card for card in catalog["cards"]},
        {attack["attackId"]: attack for attack in catalog["attacks"]},
    )


def load_deck(root: Path, cards: dict[int, CardData]) -> list[int]:
    with (root / "deck.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    deck = [int(row["card_id"]) for row in rows for _ in range(int(row["count"]))]
    validate_deck(deck, cards)
    return deck


def validate_deck(deck: list[int], cards: dict[int, CardData]) -> None:
    if len(deck) != 60:
        raise ValueError("The deck must contain exactly 60 cards")
    names: Counter[str] = Counter()
    basics = 0
    ace_specs = 0
    for card_id in deck:
        if card_id not in cards:
            raise ValueError(f"Unknown card ID: {card_id}")
        card = cards[card_id]
        basics += int(card["basic"])
        ace_specs += int(card["aceSpec"])
        if card["cardType"] != 5:
            names[card["name"]] += 1
    if basics == 0:
        raise ValueError("The deck must contain a Basic Pokémon")
    if ace_specs > 1:
        raise ValueError("The deck may contain only one ACE SPEC")
    if any(count > 4 for count in names.values()):
        raise ValueError("More than four copies of a card name")
