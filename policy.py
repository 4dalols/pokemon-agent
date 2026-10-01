import re
from collections import Counter
from math import comb, inf

from schema import AttackData, Card, CardData, Current, Observation, Option, Player, Selection

DRAW_TEXT = re.compile(r"\b[Dd]raw\b")
SELF_DAMAGE = re.compile(r"does (\d+) damage to itself")
DISCARD_OWN_ENERGY = re.compile(r"Discard (\d+|all) Energy from this Pokémon")
RECOVER_ENERGY = re.compile(r"[Aa]ttach up to (\d+) Basic \{(\w)\} Energy cards? from your discard")
IMMUNITY = re.compile(
    r"Prevent all damage done to this Pokémon by attacks from your opponent’s Pokémon \{ex\}"
)
PRIZE_SCALING = re.compile(r"(\d+) more damage .* for each Prize card your opponent has taken")
COUNTER_SCALING = re.compile(r"(\d+) more damage for each damage counter on this Pokémon")
SEARCH_ATTACH = re.compile(r"Search your deck for a Basic \{\w\} Energy card and attach")
ENERGY_SYMBOLS = {"G": 1, "R": 2, "W": 3, "L": 4, "P": 5, "F": 6, "D": 7, "M": 8}


class Policy:
    def __init__(
        self, deck: list[int], cards: dict[int, CardData], attacks: dict[int, AttackData]
    ) -> None:
        self.deck = deck
        self.cards = cards
        self.attacks = attacks
        energy_types = Counter(
            cards[card_id]["energyType"] for card_id in deck if cards[card_id]["cardType"] == 5
        )
        self.energy_type = energy_types.most_common(1)[0][0] if energy_types else 0
        self.deck_names = Counter(cards[card_id]["name"] for card_id in deck)
        self.energy_ids = {
            card_id
            for card_id, data in cards.items()
            if data["cardType"] == 5 and data["energyType"] == self.energy_type
        }
        self.deck_energy = sum(card_id in self.energy_ids for card_id in deck)
        self.evolutions: dict[str, set[str]] = {}
        for card_id in set(deck):
            parent = cards[card_id]["evolvesFrom"]
            if parent is not None and parent in self.deck_names:
                self.evolutions.setdefault(parent, set()).add(cards[card_id]["name"])
        self.power = {
            cards[card_id]["name"]: self.line_power(cards[card_id]["name"])
            for card_id in set(deck)
            if cards[card_id]["cardType"] == 0
        }

    def choose(self, observation: Observation) -> list[int]:
        selection = observation["select"]
        current = observation["current"]
        if selection is None:
            return self.deck.copy()
        options = selection["option"]
        if current is None or not options:
            return list(range(min(selection["minCount"], len(options))))
        scores = self.scores(selection, current)
        ranked = self.rank(scores)
        if selection["type"] == 4:
            return self.pay_energy(ranked, options, selection)
        count = min(selection["maxCount"], len(options))
        if selection["minCount"] == 0:
            count = min(count, sum(score > 0 for score in scores))
        return ranked[: max(selection["minCount"], count)]

    @staticmethod
    def player_index(option: Option, current: Current) -> int:
        index = option.get("playerIndex")
        return current["yourIndex"] if index is None else index

    def scores(self, selection: Selection, current: Current) -> list[float]:
        return [self.score(option, selection, current) for option in selection["option"]]

    @staticmethod
    def rank(scores: list[float]) -> list[int]:
        return sorted(range(len(scores)), key=lambda index: (-scores[index], index))

    @staticmethod
    def pay_energy(ranked: list[int], options: list[Option], selection: Selection) -> list[int]:
        required = selection["remainEnergyCost"]
        affordable = [index for index in ranked if options[index].get("count", 1) <= required]
        count = max(selection["minCount"], min(1, selection["maxCount"]))
        return (affordable if len(affordable) >= count else ranked)[:count]

    @staticmethod
    def own(current: Current) -> Player:
        return current["players"][current["yourIndex"]]

    @staticmethod
    def active(player: Player) -> Card | None:
        return player["active"][0] if player["active"] else None

    @staticmethod
    def in_play(player: Player) -> list[Card]:
        return [card for card in player["active"] + player["bench"] if card is not None]

    def resolve(self, option: Option, selection: Selection, current: Current) -> Card | None:
        area = option.get("area", 2)
        index = option.get("index", 0)
        player = current["players"][self.player_index(option, current)]
        zones: dict[int, list[Card | None]] = {
            1: list(selection["deck"] or []),
            2: list(player["hand"] or []),
            3: list(player["discard"]),
            4: player["active"],
            5: player["bench"],
            6: player["prize"],
            7: list(current["stadium"]),
            12: current["looking"] or [],
        }
        zone = zones.get(area, [])
        card = zone[index] if 0 <= index < len(zone) else None
        if card is not None and option["type"] in (4, 5, 6):
            attachments = (
                card.get("tools", []) if option["type"] == 4 else card.get("energyCards", [])
            )
            position = (
                option.get("toolIndex", 0) if option["type"] == 4 else option.get("energyIndex", 0)
            )
            return attachments[position] if position < len(attachments) else None
        return card

    def target(self, option: Option, current: Current) -> Card | None:
        player = self.own(current)
        zone = player["active"] if option.get("inPlayArea") == 4 else player["bench"]
        index = option.get("inPlayIndex", 0)
        return zone[index] if index < len(zone) else None

    # Deck structure -------------------------------------------------------------------------

    def line_power(self, name: str) -> float:
        """Best raw attack damage reachable from a card name through its deck evolutions."""
        best = 0.0
        for card_id in set(self.deck):
            data = self.cards[card_id]
            if data["name"] == name:
                best = max(
                    best,
                    max((self.base_damage(self.attacks[a]) for a in data["attacks"]), default=0.0),
                )
        for child in self.evolutions.get(name, ()):
            best = max(best, self.line_power(child))
        return best

    @staticmethod
    def base_damage(attack: AttackData) -> float:
        if attack["name"] == "Hammer-lanche":
            return 300.0
        if attack["name"] == "Riptide":
            return 120.0
        return float(attack["damage"])

    def line_count(self, name: str, names: Counter[str]) -> int:
        return names[name] + sum(
            self.line_count(child, names) for child in self.evolutions.get(name, ())
        )

    def ancestors(self, name: str) -> set[str]:
        found = {parent for parent, children in self.evolutions.items() if name in children}
        for parent in list(found):
            found |= self.ancestors(parent)
        return found

    def descendants(self, name: str) -> set[str]:
        found = set(self.evolutions.get(name, ()))
        for child in list(found):
            found |= self.descendants(child)
        return found

    def best_line(self, name: str) -> float:
        return max((self.power[child] for child in self.evolutions.get(name, ())), default=0.0)

    def pokemon_value(self, data: CardData, names: Counter[str]) -> float:
        name = data["name"]
        parent = data["evolvesFrom"]
        if parent is not None and not data["basic"]:
            return 140 if any(names[ancestor] for ancestor in self.ancestors(name)) else 35
        if name in self.evolutions:
            return 125 if self.line_count(name, names) < 2 else 15
        bonus = self.power.get(name, 0.0) / 50
        return (60 if names[name] < 2 else 12) + bonus

    # Energy and attacks ---------------------------------------------------------------------

    @staticmethod
    def missing_energy(attached: list[int], attack: AttackData) -> int:
        pool = Counter(attached)
        missing = 0
        colorless = 0
        for cost in attack["energies"]:
            if cost == 0:
                colorless += 1
            elif pool[cost] > 0:
                pool[cost] -= 1
            else:
                missing += 1
        return missing + max(0, colorless - sum(pool.values()))

    def estimate(self, attack: AttackData, card: Card, current: Current) -> float:
        name = attack["name"]
        if name == "Hammer-lanche":
            water, unseen = self.unseen_energy(current)
            return 100.0 * min(6, self.own(current)["deckCount"]) * water / max(1, unseen)
        if name == "Riptide":
            return 20.0 * self.energy_discard(current)
        damage = float(attack["damage"])
        scaling = COUNTER_SCALING.search(attack["text"])
        if scaling:
            damage += int(scaling.group(1)) * (card.get("maxHp", 0) - card.get("hp", 0)) / 10
        data = self.cards[card["id"]]
        for skill in data["skills"]:
            prizes = PRIZE_SCALING.search(skill["text"])
            if prizes:
                remaining = current["players"][1 - current["yourIndex"]]["prize"]
                damage += int(prizes.group(1)) * (6 - len(remaining) if remaining else 0)
        return damage

    def best_attack(self, card: Card, current: Current) -> tuple[AttackData | None, int]:
        """Strongest attack of a Pokémon and how many Energy it still lacks for it."""
        data = self.cards[card["id"]]
        attached = card.get("energies", [])
        best: AttackData | None = None
        best_key = (-inf, 0)
        for attack_id in self.line_attacks(data["name"]):
            attack = self.attacks[attack_id]
            missing = self.missing_energy(attached, attack)
            key = (self.estimate(attack, card, current) - 40 * missing, -missing)
            if key > best_key:
                best, best_key = attack, key
        return best, -best_key[1]

    def line_attacks(self, name: str) -> list[int]:
        """Attacks of a card plus, for Pokémon that evolve in this deck, of its evolutions."""
        attacks: list[int] = []
        for card_id in sorted(set(self.deck)):
            if self.cards[card_id]["name"] == name:
                attacks.extend(self.cards[card_id]["attacks"])
        for child in sorted(self.evolutions.get(name, ())):
            attacks.extend(self.line_attacks(child))
        return attacks

    def needed(self, card: Card, current: Current) -> int:
        return self.best_attack(card, current)[1]

    def value(self, card: Card | None, current: Current) -> float:
        if card is None or card["id"] not in self.cards:
            return 0
        data = self.cards[card["id"]]
        player = self.own(current)
        in_play = self.in_play(player)
        names = Counter(self.cards[entry["id"]]["name"] for entry in in_play)
        name = data["name"]
        kind = data["cardType"]
        text = " ".join(skill["text"] for skill in data["skills"])
        if kind == 0:
            return self.pokemon_value(data, names)
        if kind == 5:
            needs = any(self.needed(entry, current) > 0 for entry in in_play)
            return 75 if needs and not current["energyAttached"] else 8
        if kind == 6:
            return 40 if not current["energyAttached"] else 8
        if kind == 3:
            if current["supporterPlayed"]:
                return 5
            if DRAW_TEXT.search(text):
                return 100 if player["handCount"] < 5 else 20
            return 60
        if kind == 2:
            active = self.active(player)
            return 50 if active and not active.get("tools") else 5
        if kind == 4:
            return 30 if not current["stadium"] else 5
        if name == "Mega Signal" or "Mega Evolution" in text:
            return 90 if self.wants_evolution(player, names, mega=True) else 5
        if "Search your deck" in text and "Pokémon" in text:
            if "discard 2 other cards" in text and player["handCount"] <= 4:
                return 5
            needs_basic = len(in_play) < 3 and any(
                self.cards[card_id]["basic"] and self.deck_names[self.cards[card_id]["name"]]
                for card_id in set(self.deck)
            )
            return 65 if needs_basic or self.wants_evolution(player, names, mega=False) else 5
        if "from your discard pile" in text:
            discarded = [self.cards[entry["id"]] for entry in player["discard"]]
            useful = any(
                entry["cardType"] == 0 or (entry["cardType"] == 5 and not self.has_type(player, 5))
                for entry in discarded
            )
            return 60 if useful else 5
        return 20

    def wants_evolution(self, player: Player, names: Counter[str], mega: bool) -> bool:
        for parent, children in self.evolutions.items():
            if not names[parent]:
                continue
            for child in children:
                is_mega = any(
                    self.cards[card_id]["megaEx"]
                    for card_id in set(self.deck)
                    if self.cards[card_id]["name"] == child
                )
                if (is_mega or not mega) and not self.has_hand(player, child):
                    return True
        return False

    def useful_hand(self, player: Player, except_serial: int) -> bool:
        """Whether the hand holds anything playable (not Energy) besides the given card."""
        return any(
            self.cards[card["id"]]["cardType"] != 5 and card["serial"] != except_serial
            for card in player["hand"] or []
        )

    def has_hand(self, player: Player, name: str) -> bool:
        return any(self.cards[card["id"]]["name"] == name for card in player["hand"] or [])

    def has_type(self, player: Player, card_type: int) -> bool:
        return any(self.cards[card["id"]]["cardType"] == card_type for card in player["hand"] or [])

    def readiness(self, card: Card | None, current: Current) -> float:
        if card is None:
            return 0
        data = self.cards[card["id"]]
        energies = card.get("energies", [])
        score = len(energies) * 35 + card.get("hp", data["hp"]) / 10
        attack, missing = self.best_attack(card, current)
        if attack is None:
            return score
        if data["name"] == "Kyogre":
            return score + (150 if energies and self.energy_discard(current) >= 6 else 0)
        if data["name"] in self.evolutions and self.best_line(data["name"]) > self.power.get(
            data["name"], 0.0
        ):
            evolution_in_hand = any(
                self.has_hand(self.own(current), child) for child in self.descendants(data["name"])
            )
            return score + (80 if evolution_in_hand else 30)
        if missing == 0:
            return score + 200 + self.estimate(attack, card, current) / 10
        return score + (70 if missing == 1 else 0)

    def energy_discard(self, current: Current) -> int:
        return sum(
            self.cards[card["id"]]["cardType"] == 5
            and self.cards[card["id"]]["energyType"] == self.energy_type
            for card in self.own(current)["discard"]
        )

    def unseen_energy(self, current: Current) -> tuple[int, int]:
        player = self.own(current)
        visible = list(player["hand"] or []) + player["discard"]
        for card in player["active"] + player["bench"]:
            if card is not None:
                visible += [card] + card.get("energyCards", []) + card.get("tools", [])
                visible += card.get("preEvolution", [])
        visible += [card for card in player["prize"] if card is not None]
        energy = self.deck_energy - sum(card["id"] in self.energy_ids for card in visible)
        total = player["deckCount"] + sum(card is None for card in player["prize"])
        return max(0, min(energy, total)), total

    def attack_penalty(self, attack: AttackData, current: Current) -> float:
        text = attack["text"]
        penalty = 0.0
        if attack["name"] == "Hammer-lanche":
            penalty += 25 if self.own(current)["deckCount"] <= 12 else 0
        if attack["name"] == "Riptide":
            penalty += 15
        self_damage = SELF_DAMAGE.search(text)
        if self_damage:
            penalty += int(self_damage.group(1)) / 4
        discard = DISCARD_OWN_ENERGY.search(text)
        if discard:
            penalty += 20 * (3 if discard.group(1) == "all" else int(discard.group(1)))
        if "this attack does nothing" in text:
            penalty += 100
        if "can’t use" in text or "can't use" in text:
            penalty += 5
        recover = RECOVER_ENERGY.search(text)
        if recover:
            available = sum(
                self.cards[card["id"]]["cardType"] == 5
                and self.cards[card["id"]]["energyType"] == ENERGY_SYMBOLS.get(recover.group(2), 0)
                for card in self.own(current)["discard"]
            )
            penalty -= 30 * min(int(recover.group(1)), available)
        if SEARCH_ATTACH.search(text):
            penalty -= 30
        return penalty

    def attack_score(self, attack_id: int, current: Current) -> float:
        attack = self.attacks[attack_id]
        player = self.own(current)
        active = self.active(player)
        opponent = current["players"][1 - current["yourIndex"]]
        defending = self.active(opponent)
        damage = self.estimate(attack, active, current) if active else float(attack["damage"])
        penalty = self.attack_penalty(attack, current)
        multiplier = 1
        resistance = 0
        if active and defending:
            attacking_type = self.cards[active["id"]]["energyType"]
            target = self.cards[defending["id"]]
            multiplier = 2 if target["weakness"] == attacking_type else 1
            resistance = 30 if target["resistance"] == attacking_type else 0
        effective = max(0, damage * multiplier - resistance)
        if active and defending and self.immune(defending, active):
            effective = 0
        hp = defending.get("hp", 0) if defending else 0
        ko_probability = float(hp > 0 and effective >= hp)
        water, unseen = self.unseen_energy(current)
        if attack["name"] == "Hammer-lanche" and unseen:
            draws = min(6, player["deckCount"])
            needed = max(1, (hp + resistance + 100 * multiplier - 1) // (100 * multiplier))
            ko_probability = sum(
                comb(water, successes) * comb(unseen - water, draws - successes)
                for successes in range(needed, min(draws, water) + 1)
                if 0 <= draws - successes <= unseen - water
            ) / comb(unseen, draws)
            if player["deckCount"] <= 6:
                target_prizes = 1
                if defending:
                    data = self.cards[defending["id"]]
                    target_prizes = 3 if data["megaEx"] else 2 if data["ex"] else 1
                penalty += 450 * (1 - ko_probability * float(len(player["prize"]) <= target_prizes))
        prizes = 1
        if defending:
            target = self.cards[defending["id"]]
            prizes = 3 if target["megaEx"] else 2 if target["ex"] else 1
        if len(player["prize"]) <= prizes and ko_probability >= 0.99:
            return 10000 + effective
        return 65 + min(effective, hp or effective) + 150 * prizes * ko_probability - penalty

    def attach_worth(self, target: Card, energy_type: int, current: Current) -> float:
        """How much attaching one Energy of the given type helps a Pokémon in play."""
        attack, before = self.best_attack(target, current)
        if attack is None or before == 0:
            return -120
        after = self.missing_energy(target.get("energies", []) + [energy_type], attack)
        if after >= before:
            return -120
        power = self.estimate(attack, target, current)
        bonus = min(180.0, power * 1.5) if after == 0 else 0.0
        return 300 + bonus + power / 10 - len(target.get("energies", [])) * 10

    def best_damage(self, current: Current) -> float:
        active = self.active(self.own(current))
        if active is None:
            return 0
        attack, missing = self.best_attack(active, current)
        if attack is None or missing:
            return 0
        attacking_type = self.cards[active["id"]]["energyType"]
        damage = self.estimate(attack, active, current)
        opponent = current["players"][1 - current["yourIndex"]]
        defending = self.active(opponent)
        if defending and self.immune(defending, active):
            return 0
        if defending and self.cards[defending["id"]]["weakness"] == attacking_type:
            damage *= 2
        return damage

    def immune(self, defender: Card, attacker: Card) -> bool:
        """Whether the defender's Ability blocks all attack damage from this attacker."""
        attacker_data = self.cards[attacker["id"]]
        if not (attacker_data["ex"] or attacker_data["megaEx"]):
            return False
        return any(IMMUNITY.search(skill["text"]) for skill in self.cards[defender["id"]]["skills"])

    def gust_worth(self, current: Current) -> float:
        opponent = current["players"][1 - current["yourIndex"]]
        defending = self.active(opponent)
        damage = self.best_damage(current)
        benched = [card for card in opponent["bench"] if card is not None]
        if not damage or defending is None or not benched:
            return -120
        if defending.get("hp", 0) <= damage:
            return -120
        weakest = min(card.get("hp", 0) for card in benched)
        return 220 if weakest <= damage else -120

    def score(self, option: Option, selection: Selection, current: Current) -> float:
        kind = option["type"]
        player = self.own(current)
        active = self.active(player)
        card = self.resolve(option, selection, current)
        context = selection["context"]
        if kind == 14:
            return -100
        if kind == 13:
            score = self.attack_score(option["attackId"], current)
            return score / 10 if selection["type"] == 0 else score
        if kind == 9:
            return 500 + (100 if option.get("inPlayArea") == 4 else 0)
        if kind == 8:
            target = self.target(option, current)
            if target is None or card is None:
                return 0
            data = self.cards[card["id"]]
            if data["cardType"] in (5, 6):
                worth = self.attach_worth(target, data["energyType"], current)
                return worth + (80 if option.get("inPlayArea") == 4 and worth > 0 else 0)
            if option.get("inPlayArea") == 4 and self.needed(target, current) <= 1:
                return 340
            return 50
        if kind == 7 and card is not None:
            data = self.cards[card["id"]]
            name = data["name"]
            text = " ".join(skill["text"] for skill in data["skills"])
            if data["cardType"] == 3 and DRAW_TEXT.search(text):
                if player["handCount"] <= 5 or not self.useful_hand(player, card["serial"]):
                    return 360
                return -120 if "huffle your hand" in text else 140
            if name == "Secret Box":
                return 190 if player["handCount"] >= 6 else -120
            if name == "Surfing Beach" or "Switch your Active Pokémon" in text:
                return (
                    160 if self.best_bench(current) > self.readiness(active, current) + 40 else -120
                )
            if "opponent’s Benched Pokémon to the Active Spot" in text:
                return self.gust_worth(current)
            if data["basic"]:
                return self.value(card, current) * 3 if len(player["bench"]) < 3 else -120
            return self.value(card, current) * 3
        if kind == 10 and card is not None and self.cards[card["id"]]["cardType"] == 0:
            return 450
        if kind in (10, 12):
            improvement = self.best_bench(current) - self.readiness(active, current)
            return 350 + improvement if improvement > 40 else -120
        if kind == 11:
            return -150
        if kind == 0:
            return float(option.get("number", 0)) if context == 38 else 0
        if kind in (1, 2):
            return float(kind == 1)
        if kind in (3, 4, 5, 6):
            if context in (8, 9, 10, 11, 26, 27, 29, 30, 31, 32):
                return 200 - self.value(card, current)
            if context == 1:
                return self.setup_value(card)
            if context in (3, 4):
                if self.player_index(option, current) != current["yourIndex"]:
                    return 400 - (card.get("hp", 0) if card else 0)
                return self.readiness(card, current)
            if context in (13, 14, 15):
                return 400 - (card.get("hp", 0) if card else 0)
            if context in (16, 17):
                return float(card.get("maxHp", 0) - card.get("hp", 0)) if card else 0
            if context == 22 and card is not None and self.cards[card["id"]]["cardType"] == 0:
                return max(1.0, self.attach_worth(card, self.energy_type, current))
            if context in (18, 21):
                return self.readiness(card, current)
            return self.value(card, current)
        return 0

    def setup_value(self, card: Card | None) -> float:
        if card is None or card["id"] not in self.cards:
            return 0
        data = self.cards[card["id"]]
        if data["cardType"] != 0:
            return 0
        return self.pokemon_value(data, Counter())

    def best_bench(self, current: Current) -> float:
        return max(
            (self.readiness(card, current) for card in self.own(current)["bench"]), default=0
        )
