"""Imitation (behaviour-cloning) ranker learned from public Playground replays.

Every option of a selection is described by sparse string features: the selection
kind, the option itself (card, target, attack), the visible game state and the
selections already made this turn. A linear model over those features scores the
options; `BCPolicy` plugs that score into `Policy.choose`, so the same ranking rules
(option positions, iterative energy payments, optional picks) apply. Weights are
trained offline by `train_bc.py` and shipped as JSON; inference is pure Python.
"""

import json
from collections import Counter
from pathlib import Path
from typing import TypedDict, cast

from policy import Policy
from schema import AttackData, Card, CardData, Current, Observation, Option, Selection

MODEL_FILE = "bc_model.json"
HISTORY_LIMIT = 12


class Model(TypedDict):
    weights: dict[str, float]
    bias: dict[str, float]


def bucket(value: int, edges: tuple[int, ...]) -> int:
    return sum(value >= edge for edge in edges)


HP_EDGES = (1, 61, 121, 181, 241, 301)
COUNT_EDGES = (1, 2, 3, 4, 6)
HAND_EDGES = (1, 3, 5, 7, 10)
TURN_EDGES = (2, 3, 5, 9, 15)
DECK_EDGES = (1, 7, 13, 25, 40)


class Featurizer:
    def __init__(self, cards: dict[int, CardData], attacks: dict[int, AttackData]) -> None:
        self.cards = cards
        self.attacks = attacks
        self.policy = Policy([], cards, attacks)

    def describe(self, card: Card | None) -> list[str]:
        if card is None:
            return ["c=?"]
        data = self.cards.get(card["id"])
        if data is None:
            return [f"c={card['id']}"]
        out = [f"c={card['id']}", f"ct={data['cardType']}"]
        if data["cardType"] == 0:
            stage = 2 if data["stage2"] else 1 if data["stage1"] else 0
            out.append(f"stage={stage}{'x' if data['ex'] or data['megaEx'] else ''}")
            hp = card.get("hp")
            if hp is not None:
                out.append(f"hp={bucket(hp, HP_EDGES)}")
                out.append(f"dmg={bucket(card.get('maxHp', hp) - hp, HP_EDGES)}")
            energies = card.get("energies")
            if energies is not None:
                out.append(f"en={bucket(len(energies), COUNT_EDGES)}")
        return out

    def context(self, selection: Selection, current: Current, history: list[str]) -> list[str]:
        me = current["yourIndex"]
        mine, theirs = current["players"][me], current["players"][1 - me]
        active, enemy = Policy.active(mine), Policy.active(theirs)
        out = [
            f"turn={bucket(current['turn'], TURN_EDGES)}",
            f"first={int(current['turn'] <= 2 and current.get('firstPlayer', -1) == me)}",
            f"hand={bucket(mine['handCount'], HAND_EDGES)}",
            f"bench={len([card for card in mine['bench'] if card is not None])}",
            f"obench={len([card for card in theirs['bench'] if card is not None])}",
            f"prize={len(mine['prize'])}",
            f"oprize={len(theirs['prize'])}",
            f"deck={bucket(mine['deckCount'], DECK_EDGES)}",
            f"ea={int(current['energyAttached'])}",
            f"sp={int(current['supporterPlayed'])}",
            f"acts={min(current['turnActionCount'], 6)}",
            f"ctx={selection['context']}",
        ]
        out.extend(f"act:{tag}" for tag in self.describe(active))
        out.extend(f"oact:{tag}" for tag in self.describe(enemy))
        if current["stadium"]:
            out.append(f"stad={current['stadium'][0]['id']}")
        hand = Counter(card["id"] for card in mine["hand"] or [])
        out.extend(f"hand:{card_id}" for card_id in hand)
        out.extend(f"hand2:{card_id}" for card_id, count in hand.items() if count > 1)
        out.extend(f"mine:{card['id']}" for card in mine["bench"] if card is not None)
        out.extend(f"opp:{card['id']}" for card in theirs["bench"] if card is not None)
        out.extend(f"h:{tag}" for tag in history)
        if history:
            out.append(f"hlast:{history[-1]}")
        else:
            out.append("h:none")
        return out

    def option(self, option: Option, selection: Selection, current: Current) -> list[str]:
        kind = option["type"]
        card = self.policy.resolve(option, selection, current)
        card_tags = self.describe(card)
        head = f"o={kind}"
        if kind in (3, 4, 5, 6):
            owner = Policy.player_index(option, current)
            out = [f"{head}:{tag}" for tag in card_tags]
            own = int(owner == current["yourIndex"])
            out.append(f"{head}|area={option.get('area', 0)}|own={own}")
            if card is not None and option.get("area") == 2:
                hand = current["players"][owner]["hand"] or []
                copies = sum(entry["id"] == card["id"] for entry in hand)
                out.append(f"{head}|copies={min(copies, 3)}")
            if card is not None and option.get("area") in (4, 5):
                out.append(f"{head}|inplay={option.get('area')}")
            if kind == 6:
                out.append(f"{head}|cnt={option.get('count', 1)}")
            return out
        if kind == 7:
            return [f"{head}:{tag}" for tag in card_tags]
        if kind in (8, 9):
            target = self.policy.target(option, current)
            target_tags = self.describe(target)
            out = [f"{head}:{tag}" for tag in card_tags]
            out.extend(f"{head}>{tag}" for tag in target_tags)
            out.append(f"{head}|tact={int(option.get('inPlayArea') == 4)}")
            if card is not None and target is not None:
                out.append(f"{head}:{card['id']}>{target['id']}")
                if kind == 8:
                    level = bucket(len(target.get("energies", [])), COUNT_EDGES)
                    out.append(f"{head}:{card['id']}>en={level}")
            return out
        if kind == 10:
            return [f"{head}:{tag}" for tag in card_tags] + [f"{head}|area={option.get('area', 0)}"]
        if kind == 12:
            mine = Policy.own(current)
            active = Policy.active(mine)
            out = [head]
            out.extend(f"{head}:{tag}" for tag in self.describe(active))
            best = max((card.get("hp", 0) for card in mine["bench"] if card is not None), default=0)
            out.append(f"{head}|benchhp={bucket(best, HP_EDGES)}")
            return out
        if kind == 13:
            return self.attack(option.get("attackId", -1), current)
        if kind == 0:
            numbers = [entry.get("number", 0) for entry in selection["option"]]
            number = option.get("number", 0)
            return [
                f"{head}:{number}",
                f"{head}|max={int(number == max(numbers))}",
                f"{head}|min={int(number == min(numbers))}",
            ]
        if kind == 15:
            return [f"{head}:{option.get('cardId', -1)}"]
        return [head]

    def attack(self, attack_id: int, current: Current) -> list[str]:
        head = "o=13"
        attack = self.attacks.get(attack_id)
        out = [f"{head}:{attack_id}"]
        if attack is None:
            return out
        me = current["yourIndex"]
        active = Policy.active(current["players"][me])
        enemy = Policy.active(current["players"][1 - me])
        damage = attack["damage"]
        out.append(f"{head}|dmg={bucket(damage, HP_EDGES)}")
        if active is not None:
            out.append(f"{head}:{attack_id}|from={active['id']}")
        if enemy is not None and enemy.get("hp") is not None:
            hp = enemy.get("hp", 0)
            data = self.cards.get(enemy["id"])
            attacker = self.cards.get(active["id"]) if active is not None else None
            effective = damage
            if data is not None and attacker is not None:
                if data["weakness"] == attacker["energyType"]:
                    effective *= 2
                if data["resistance"] == attacker["energyType"]:
                    effective = max(0, effective - 30)
            out.append(f"{head}|ko={int(effective >= hp > 0)}")
            out.append(f"{head}|margin={bucket(max(0, hp - effective), HP_EDGES)}")
            out.append(f"{head}:{attack_id}|vs={enemy['id']}")
        return out

    def features(
        self, selection: Selection, current: Current, history: list[str]
    ) -> list[list[str]]:
        """One sparse row per option, plus a trailing "decline" row for optional picks."""
        kind = f"s={selection['type']}"
        context = self.context(selection, current, history)
        rows: list[list[str]] = []
        for position, option in enumerate(selection["option"]):
            tags = self.option(option, selection, current)
            primary = tags[0]
            row = [f"{kind}|{tag}" for tag in tags]
            row.append(f"{kind}|pos={min(position, 8)}")
            row.append(f"{kind}|{primary}|pos={min(position, 8)}")
            row.extend(f"{kind}|{primary}|{tag}" for tag in context)
            if len(tags) > 1:
                row.extend(f"{kind}|{tags[1]}|{tag}" for tag in context[:12])
            rows.append(row)
        if selection["minCount"] == 0:
            decline = f"{kind}|decline"
            row = [decline, f"{decline}|ctx={selection['context']}"]
            row.extend(f"{decline}|{tag}" for tag in context)
            rows.append(row)
        return rows


def selection_history(selection: Selection, current: Current, action: list[int]) -> list[str]:
    """Descriptors of the selected options, appended to the same-turn history."""
    options = selection["option"]
    out: list[str] = []
    featurizer_policy = Policy([], {}, {})
    for index in action:
        if not 0 <= index < len(options):
            continue
        option = options[index]
        kind = option["type"]
        if kind == 13:
            out.append(f"{selection['type']}:atk{option.get('attackId', -1)}")
        elif kind in (0, 1, 2, 12, 14):
            out.append(f"{selection['type']}:{selection['context']}:{kind}")
        else:
            card = featurizer_policy.resolve(option, selection, current)
            card_id = card["id"] if card is not None else -1
            out.append(f"{selection['type']}:{selection['context']}:{kind}:{card_id}")
    return out


def load_model(root: Path) -> Model | None:
    path = root / MODEL_FILE
    if not path.exists():
        path = Path("/kaggle_simulations/agent") / MODEL_FILE
    if not path.exists():
        return None
    return cast(Model, json.loads(path.read_text(encoding="utf-8")))


class BCPolicy(Policy):
    """Policy whose option scores come from the learned ranker.

    The same-turn history is kept per (turn, player) and reset when either changes;
    `snapshot`/`restore` let the searcher branch without leaking rollout history.
    `types` restricts the learned scores to those selection types (heuristic otherwise).
    """

    def __init__(
        self,
        deck: list[int],
        cards: dict[int, CardData],
        attacks: dict[int, AttackData],
        model: Model,
        types: frozenset[int] | None = None,
    ) -> None:
        super().__init__(deck, cards, attacks)
        self.types = types
        self.weights = model["weights"]
        self.bias = model["bias"]
        self.featurizer = Featurizer(cards, attacks)
        self.turn_key: tuple[int, int] = (-1, -1)
        self.history: list[str] = []

    def observe(self, observation: Observation, action: list[int]) -> None:
        selection, current = observation["select"], observation["current"]
        if selection is None or current is None:
            return
        self.sync(current)
        self.history.extend(selection_history(selection, current, action))
        del self.history[:-HISTORY_LIMIT]

    def sync(self, current: Current) -> None:
        key = (current["turn"], current["yourIndex"])
        if key != self.turn_key:
            self.turn_key = key
            self.history = []

    def snapshot(self) -> tuple[tuple[int, int], list[str]]:
        return self.turn_key, list(self.history)

    def restore(self, state: tuple[tuple[int, int], list[str]]) -> None:
        self.turn_key, self.history = state[0], list(state[1])

    def scores(self, selection: Selection, current: Current) -> list[float]:
        """Option scores; for optional picks they are relative to declining (> 0 = take)."""
        self.sync(current)
        if self.types is not None and selection["type"] not in self.types:
            return super().scores(selection, current)
        rows = self.featurizer.features(selection, current, self.history)
        weights = self.weights
        scores = [sum(weights.get(tag, 0.0) for tag in row) for row in rows]
        if selection["minCount"] == 0:
            decline = scores.pop()
            return [score - decline for score in scores]
        return scores
