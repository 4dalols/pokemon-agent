import argparse
import ctypes
import json
from pathlib import Path
from typing import cast

from assets import validate_deck
from decks import DECKS, DEFAULT_DECK, deck_list
from engine import SOURCE, lib
from schema import AttackData, CardData, Catalog

ROOT = Path(__file__).resolve().parent


def prepare(deck_name: str = DEFAULT_DECK) -> None:
    lib.AllCard.argtypes = []
    lib.AllCard.restype = ctypes.c_char_p
    lib.AllAttack.argtypes = []
    lib.AllAttack.restype = ctypes.c_char_p
    cards = cast(list[CardData], json.loads(lib.AllCard()))
    attacks = cast(list[AttackData], json.loads(lib.AllAttack()))
    catalog: Catalog = {"source": SOURCE, "cards": cards, "attacks": attacks}
    deck = deck_list(deck_name)
    validate_deck(deck, {card["cardId"]: card for card in cards})
    (ROOT / "cards.json").write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    (ROOT / "deck.csv").write_text("\n".join(str(card_id) for card_id in deck) + "\n")
    print(
        f"Prepared {len(cards)} cards and {len(attacks)} attacks from {SOURCE}; "
        f"wrote 60-card deck {deck_name!r}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", choices=sorted(DECKS), default=DEFAULT_DECK)
    prepare(parser.parse_args().deck)
