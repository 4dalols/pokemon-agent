import ctypes
import json
from pathlib import Path
from typing import cast

from assets import validate_deck
from engine import SOURCE, lib
from schema import AttackData, CardData, Catalog

ROOT = Path(__file__).resolve().parent
DECK = {
    721: 2,
    722: 4,
    723: 4,
    1092: 1,
    1121: 2,
    1145: 2,
    1163: 2,
    1219: 4,
    1227: 4,
    1262: 2,
    3: 33,
}


def prepare() -> None:
    lib.AllCard.argtypes = []
    lib.AllCard.restype = ctypes.c_char_p
    lib.AllAttack.argtypes = []
    lib.AllAttack.restype = ctypes.c_char_p
    cards = cast(list[CardData], json.loads(lib.AllCard()))
    attacks = cast(list[AttackData], json.loads(lib.AllAttack()))
    catalog: Catalog = {"source": SOURCE, "cards": cards, "attacks": attacks}
    deck = [card_id for card_id, count in DECK.items() for _ in range(count)]
    validate_deck(deck, {card["cardId"]: card for card in cards})
    (ROOT / "cards.json").write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    (ROOT / "deck.csv").write_text("\n".join(str(card_id) for card_id in deck) + "\n")
    print(
        f"Prepared {len(cards)} cards and {len(attacks)} attacks from {SOURCE}; wrote 60-card deck"
    )


if __name__ == "__main__":
    prepare()
