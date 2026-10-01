import json
from collections import Counter
from pathlib import Path
from typing import cast

from schema import AttackData, CardData, Catalog, PanelEntry

ROOT = Path(__file__).resolve().parent
KAGGLE_AGENT_DIR = Path("/kaggle_simulations/agent")


def locate(root: Path, name: str) -> Path:
    path = root / name
    return path if path.exists() else KAGGLE_AGENT_DIR / name


def load_catalog(root: Path) -> tuple[dict[int, CardData], dict[int, AttackData]]:
    catalog = cast(Catalog, json.loads(locate(root, "cards.json").read_text(encoding="utf-8")))
    return (
        {card["cardId"]: card for card in catalog["cards"]},
        {attack["attackId"]: attack for attack in catalog["attacks"]},
    )


def load_deck(root: Path, cards: dict[int, CardData]) -> list[int]:
    lines = locate(root, "deck.csv").read_text().split("\n")
    deck = [int(line.strip()) for line in lines if line.strip()]
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


def load_panel(path: Path, cards: dict[int, CardData]) -> dict[str, tuple[list[int], int]]:
    """Opponent deck panel: `{name: {"entries": weight, "cards": {"<id> <name>": count}}}`."""
    panel = cast(dict[str, PanelEntry], json.loads(path.read_text(encoding="utf-8")))
    decks: dict[str, tuple[list[int], int]] = {}
    for name, entry in panel.items():
        deck: list[int] = []
        for key, count in entry["cards"].items():
            card_id = int(key.split(" ", 1)[0])
            if card_id not in cards:
                raise ValueError(f"{name}: unknown card {key}")
            deck.extend([card_id] * count)
        validate_deck(sorted(deck), cards)
        decks[name] = (sorted(deck), entry["entries"])
    return decks
