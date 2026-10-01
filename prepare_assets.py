import csv
import ctypes
import json
from collections import Counter
from pathlib import Path
from typing import cast

from kaggle_environments.envs.cabt.cabt import deck
from kaggle_environments.envs.cabt.cg.sim import lib

from assets import validate_deck
from schema import AttackData, CardData, Catalog

ROOT = Path(__file__).resolve().parent


def prepare() -> None:
    lib.AllCard.argtypes = []
    lib.AllCard.restype = ctypes.c_char_p
    lib.AllAttack.argtypes = []
    lib.AllAttack.restype = ctypes.c_char_p
    cards = cast(list[CardData], json.loads(lib.AllCard()))
    attacks = cast(list[AttackData], json.loads(lib.AllAttack()))
    catalog: Catalog = {
        "source": "kaggle-environments==1.32.7 public native engine",
        "cards": cards,
        "attacks": attacks,
    }
    validate_deck(deck, {card["cardId"]: card for card in cards})
    (ROOT / "cards.json").write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
    with (ROOT / "deck.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["card_id", "count"])
        writer.writerows(Counter(deck).items())
    print(f"Prepared {len(cards)} public cards, {len(attacks)} attacks, and a 60-card sample deck")


if __name__ == "__main__":
    prepare()
