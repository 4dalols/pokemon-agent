import argparse
import ctypes
import json
from pathlib import Path
from typing import cast

from assets import validate_deck
from engine import SOURCE, lib
from schema import AttackData, CardData, Catalog

ROOT = Path(__file__).resolve().parent
DECKS: dict[str, dict[int, int]] = {
    "abomasnow": {
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
    },
    # The list played by the top-rated Playground teams; the imitation data is mostly it.
    "dragapult": {
        2: 4,
        5: 4,
        7: 2,
        112: 2,
        119: 4,
        120: 4,
        121: 3,
        140: 1,
        235: 1,
        1071: 1,
        1080: 1,
        1086: 4,
        1097: 2,
        1120: 4,
        1121: 4,
        1152: 4,
        1182: 3,
        1197: 2,
        1198: 3,
        1227: 4,
        1231: 1,
        1246: 2,
    },
}
DEFAULT_DECK = "dragapult"


def prepare(deck_name: str = DEFAULT_DECK) -> None:
    lib.AllCard.argtypes = []
    lib.AllCard.restype = ctypes.c_char_p
    lib.AllAttack.argtypes = []
    lib.AllAttack.restype = ctypes.c_char_p
    cards = cast(list[CardData], json.loads(lib.AllCard()))
    attacks = cast(list[AttackData], json.loads(lib.AllAttack()))
    catalog: Catalog = {"source": SOURCE, "cards": cards, "attacks": attacks}
    deck = [card_id for card_id, count in DECKS[deck_name].items() for _ in range(count)]
    validate_deck(deck, {card["cardId"]: card for card in cards})
    (ROOT / "cards.json").write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    listing = "\n".join(str(card_id) for card_id in deck) + "\n"
    (ROOT / "deck.csv").write_text(listing)
    (ROOT / f"deck-{deck_name}.csv").write_text(listing)
    print(
        f"Prepared {len(cards)} cards and {len(attacks)} attacks from {SOURCE}; "
        f"wrote 60-card {deck_name} deck"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", choices=sorted(DECKS), default=DEFAULT_DECK)
    prepare(parser.parse_args().deck)
