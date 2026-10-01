"""Learned board evaluation: hand-written features and a tiny MLP in pure Python.

`featurize` turns a `Current` game state into a fixed vector from one player's point
of view; `ValueModel.predict` maps it to a win probability. Weights are trained by
`train_value.py` and shipped as `value.json`, so the runtime needs no numpy.
"""

import json
import math
from pathlib import Path
from typing import TypedDict

from schema import AttackData, Card, CardData, Current, Player

STATUSES = ("poisoned", "burned", "asleep", "paralyzed", "confused")
SIDE_FEATURES = (
    "prizes",
    "active_present",
    "active_hp",
    "active_max_hp",
    "active_damage",
    "active_energies",
    "active_matching_energies",
    "active_tool",
    "active_stage",
    "active_prize_value",
    "active_retreat",
    "active_attack_ready",
    "active_attack_damage",
    "active_best_damage",
    "active_can_knock_out",
    "active_weak",
    "active_resists",
    "bench_count",
    "bench_energies",
    "bench_hp",
    "bench_damage",
    "bench_best_energies",
    "bench_evolvable",
    "hand",
    "deck",
    "discard",
    "discard_energy",
    *STATUSES,
)
GLOBAL_FEATURES = ("turn", "to_move", "energy_attached", "supporter_played", "prize_lead")
FEATURE_NAMES = (
    *(f"my_{name}" for name in SIDE_FEATURES),
    *(f"their_{name}" for name in SIDE_FEATURES),
    *GLOBAL_FEATURES,
)


class Weights(TypedDict):
    features: list[str]
    mean: list[float]
    scale: list[float]
    layers: list[list[list[float]]]
    biases: list[list[float]]
    metadata: dict[str, float | int | str]


class Featurizer:
    def __init__(self, cards: dict[int, CardData], attacks: dict[int, AttackData]) -> None:
        self.cards = cards
        self.attacks = attacks

    def featurize(self, current: Current, me: int) -> list[float]:
        mine, theirs = current["players"][me], current["players"][1 - me]
        features = self.side(mine, theirs) + self.side(theirs, mine)
        features.extend(
            (
                min(current["turn"], 40) / 40,
                float(current["yourIndex"] == me),
                float(current["energyAttached"]),
                float(current["supporterPlayed"]),
                (len(theirs["prize"]) - len(mine["prize"])) / 6,
            )
        )
        return features

    def side(self, player: Player, opponent: Player) -> list[float]:
        active = player["active"][0] if player["active"] else None
        defending = opponent["active"][0] if opponent["active"] else None
        bench = [card for card in player["bench"] if card is not None]
        features = [len(player["prize"]) / 6]
        features.extend(self.pokemon(active, defending))
        evolving = {self.evolves_from(card) for card in player["hand"] or []}
        features.extend(
            (
                len(bench) / 5,
                sum(len(card.get("energies", [])) for card in bench) / 10,
                sum(card.get("hp", 0) for card in bench) / 1000,
                sum(card.get("maxHp", 0) - card.get("hp", 0) for card in bench) / 500,
                max((len(card.get("energies", [])) for card in bench), default=0) / 4,
                sum(self.name(card) in evolving for card in bench) / 5,
                min(player["handCount"], 12) / 12,
                player["deckCount"] / 60,
                len(player["discard"]) / 60,
                sum(self.is_energy(card) for card in player["discard"]) / 20,
            )
        )
        features.extend(float(bool(player.get(status, False))) for status in STATUSES)
        return features

    def pokemon(self, card: Card | None, defending: Card | None) -> list[float]:
        if card is None or card["id"] not in self.cards:
            return [0.0] * 16
        data = self.cards[card["id"]]
        energies = card.get("energies", [])
        hp = card.get("hp", data["hp"])
        max_hp = card.get("maxHp", data["hp"])
        ready, best, affordable = self.attack_damage(data, energies)
        target = self.cards.get(defending["id"]) if defending is not None else None
        kind = data["energyType"]
        multiplier = 2.0 if target is not None and target["weakness"] == kind else 1.0
        resistance = 30.0 if target is not None and target["resistance"] == kind else 0.0
        effective = max(0.0, affordable * multiplier - resistance)
        target_hp = defending.get("hp", 0) if defending is not None else 0
        return [
            1.0,
            hp / 350,
            max_hp / 350,
            (max_hp - hp) / 350,
            len(energies) / 4,
            energies.count(data["energyType"]) / 4,
            float(bool(card.get("tools"))),
            (2.0 if data["megaEx"] or data["stage2"] else 1.0 if data["stage1"] else 0.0) / 2,
            (3.0 if data["megaEx"] else 2.0 if data["ex"] else 1.0) / 3,
            data["retreatCost"] / 4,
            float(ready),
            effective / 300,
            best / 300,
            float(target_hp > 0 and effective >= target_hp),
            float(target is not None and data["weakness"] == target["energyType"]),
            float(target is not None and data["resistance"] == target["energyType"]),
        ]

    def attack_damage(self, data: CardData, energies: list[int]) -> tuple[bool, float, float]:
        ready = False
        best = 0.0
        affordable = 0.0
        for attack_id in data["attacks"]:
            attack = self.attacks.get(attack_id)
            if attack is None:
                continue
            damage = float(attack["damage"]) if attack["damage"] > 0 else 100.0
            best = max(best, damage)
            if self.payable(attack["energies"], energies):
                ready = True
                affordable = max(affordable, damage)
        return ready, best, affordable

    @staticmethod
    def payable(cost: list[int], energies: list[int]) -> bool:
        remaining = list(energies)
        for kind in cost:
            if kind == 0:
                continue
            if kind not in remaining:
                return False
            remaining.remove(kind)
        return len(remaining) >= cost.count(0)

    def name(self, card: Card) -> str:
        data = self.cards.get(card["id"])
        return data["name"] if data is not None else ""

    def evolves_from(self, card: Card) -> str | None:
        data = self.cards.get(card["id"])
        return data["evolvesFrom"] if data is not None else None

    def is_energy(self, card: Card) -> bool:
        data = self.cards.get(card["id"])
        return data is not None and data["cardType"] == 5


class ValueModel:
    def __init__(self, weights: Weights) -> None:
        if list(weights["features"]) != list(FEATURE_NAMES):
            raise ValueError("Weights were trained for a different feature set")
        self.mean = weights["mean"]
        self.scale = weights["scale"]
        self.layers = weights["layers"]
        self.biases = weights["biases"]

    def predict(self, features: list[float]) -> float:
        """Win probability for the side the features were built for."""
        activations = [
            (value - mean) / scale
            for value, mean, scale in zip(features, self.mean, self.scale, strict=True)
        ]
        last = len(self.layers) - 1
        for depth, (matrix, bias) in enumerate(zip(self.layers, self.biases, strict=True)):
            outputs = [
                sum(weight * value for weight, value in zip(row, activations, strict=True)) + offset
                for row, offset in zip(matrix, bias, strict=True)
            ]
            activations = outputs if depth == last else [max(0.0, value) for value in outputs]
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, activations[0]))))

    @property
    def parameters(self) -> int:
        return sum(len(row) + 1 for matrix in self.layers for row in matrix)


def load_model(path: Path) -> ValueModel | None:
    if not path.exists():
        return None
    weights: Weights = json.loads(path.read_text(encoding="utf-8"))
    return ValueModel(weights)
