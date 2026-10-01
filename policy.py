from collections import Counter
from itertools import combinations
from math import ceil, comb

from schema import AttackData, Card, CardData, Current, Observation, Option, Player, Selection


class Policy:
    def __init__(
        self, deck: list[int], cards: dict[int, CardData], attacks: dict[int, AttackData]
    ) -> None:
        self.deck = deck
        self.cards = cards
        self.attacks = attacks
        self.deck_cards = {card_id: cards[card_id] for card_id in deck}
        self.evolutions = {
            data["name"]: [
                evolution
                for evolution in self.deck_cards.values()
                if evolution["evolvesFrom"] == data["name"]
            ]
            for data in cards.values()
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
        if selection["context"] == 14:
            return self.counter_scores(selection, current)
        return [self.score(option, selection, current) for option in selection["option"]]

    def counter_scores(self, selection: Selection, current: Current) -> list[float]:
        cards = [self.resolve(option, selection, current) for option in selection["option"]]
        live = [i for i, card in enumerate(cards) if card and card.get("hp", 0) > 0]
        costs = {
            i: ceil(card.get("hp", 0) / 10) for i, card in enumerate(cards) if card and i in live
        }
        prizes = {i: self.prizes(cards[i]) for i in live}
        feasible = [
            subset
            for count in range(1, len(live) + 1)
            for subset in combinations(live, count)
            if sum(costs[i] for i in subset) <= selection["remainDamageCounter"]
        ]
        best = max(
            feasible,
            key=lambda subset: (sum(prizes[i] for i in subset), -sum(costs[i] for i in subset)),
            default=(),
        )
        return [
            (10000 if i in best else 0) - costs[i] if i in live else -100000
            for i in range(len(cards))
        ]

    def prizes(self, card: Card | None) -> int:
        if card is None:
            return 0
        data = self.cards[card["id"]]
        return 3 if data["megaEx"] else 2 if data["ex"] else 1

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

    def value(self, card: Card | None, current: Current) -> float:
        if card is None or card["id"] not in self.cards:
            return 0
        data = self.cards[card["id"]]
        player = self.own(current)
        in_play = [entry for entry in player["active"] + player["bench"] if entry is not None]
        names = Counter(self.cards[entry["id"]]["name"] for entry in in_play)
        name = data["name"]
        if name == "Snover":
            return 125 if names["Snover"] + names["Mega Abomasnow ex"] < 2 else 15
        if name == "Mega Abomasnow ex":
            return 140 if names["Snover"] else 35
        if name == "Kyogre":
            return 60 if not names["Kyogre"] else 12
        if data["cardType"] == 5:
            if 121 in self.deck_cards:
                gain = max(
                    (
                        self.energy_deficit(entry) - self.energy_deficit(entry, data["energyType"])
                        for entry in in_play
                    ),
                    default=0,
                )
                return 30 + 70 * gain
            if data["energyType"] == 1 and any(
                skill["name"] in ("Teal Dance", "Ripening Charge")
                for entry in in_play
                for skill in self.cards[entry["id"]]["skills"]
            ):
                return 90 if player["handCount"] < 6 else 70
            active = self.active(player)
            energies = len(active.get("energies", [])) if active else 0
            return 75 if energies < 2 and not current["energyAttached"] else 8
        if name == "Lillie's Determination":
            return 100 if player["handCount"] < 5 and not current["supporterPlayed"] else 20
        if name == "Mega Signal":
            return 90 if names["Snover"] and not self.has_hand(player, "Mega Abomasnow ex") else 5
        if name == "Ultra Ball":
            needs_basic = names["Snover"] + names["Mega Abomasnow ex"] < 2
            needs_evolution = names["Snover"] and not self.has_hand(player, "Mega Abomasnow ex")
            if "Snover" in {data["name"] for data in self.deck_cards.values()}:
                return 65 if needs_basic or needs_evolution else 5
            return (
                80
                if len(in_play) < 3
                or any(self.evolutions[self.cards[entry["id"]]["name"]] for entry in in_play)
                else 15
            )
        if name == "Powerglass":
            active = self.active(player)
            return 50 if active and not active.get("tools") else 5
        if name == "Team Rocket's Petrel":
            return 60 if not current["supporterPlayed"] else 5
        if data["cardType"] == 0:
            if data["evolvesFrom"]:
                return 140 if names[data["evolvesFrom"]] else 35
            if names[name] >= (3 if self.evolutions[name] or name == "Teal Mask Ogerpon ex" else 1):
                return 5
            return 100 if self.evolutions[name] or data["megaEx"] else 75
        if name in ("Rare Candy", "Dawn", "Buddy-Buddy Poffin", "Poké Pad", "Bug Catching Set"):
            return 85
        if name == "Crushing Hammer":
            return 70
        if name == "Forest of Vitality":
            return 100 if any(self.evolutions[self.cards[c["id"]]["name"]] for c in in_play) else 30
        return 20

    def energy_need(self, card: Card) -> list[int]:
        data = self.cards[card["id"]]
        if any(skill["name"] == "Adrena-Brain" for skill in data["skills"]):
            return [7]
        while self.evolutions[data["name"]]:
            data = self.evolutions[data["name"]][0]
        if data["name"] == "Slowking":
            return [5, 0]
        if not data["attacks"]:
            return []
        attack = max((self.attacks[i] for i in data["attacks"]), key=lambda a: a["damage"])
        return attack["energies"]

    def energy_deficit(self, card: Card, extra: int | None = None) -> int:
        have = Counter(card.get("energies", []))
        if extra is not None:
            have[extra] += 1
        need = Counter(self.energy_need(card))
        missing = 0
        for kind, count in need.items():
            if kind != 0:
                used = min(have[kind], count)
                have[kind] -= used
                missing += count - used
        return missing + max(0, need[0] - sum(have.values()))

    def has_hand(self, player: Player, name: str) -> bool:
        return any(self.cards[card["id"]]["name"] == name for card in player["hand"] or [])

    def readiness(self, card: Card | None, current: Current) -> float:
        if card is None:
            return 0
        data = self.cards[card["id"]]
        water = card.get("energies", []).count(3)
        score = water * 35 + card.get("hp", data["hp"]) / 10
        if data["name"] == "Mega Abomasnow ex":
            return score + (200 if water >= 2 else 70)
        if data["name"] == "Snover":
            return score + (80 if self.has_hand(self.own(current), "Mega Abomasnow ex") else 30)
        if data["name"] == "Kyogre":
            return score + (150 if water and self.water_discard(current) >= 6 else 0)
        if data["name"] in ("Munkidori", "Fezandipiti ex", "Meowth ex"):
            return card.get("hp", data["hp"]) / 10 - 50
        return (
            card.get("hp", data["hp"]) / 10
            + 200
            - 80 * self.energy_deficit(card)
            + (50 if data["ex"] or data["megaEx"] else 0)
        )

    def water_discard(self, current: Current) -> int:
        return sum(
            self.cards[card["id"]]["cardType"] == 5 and self.cards[card["id"]]["energyType"] == 3
            for card in self.own(current)["discard"]
        )

    def unseen_water(self, current: Current) -> tuple[int, int]:
        player = self.own(current)
        visible = list(player["hand"] or []) + player["discard"]
        for card in player["active"] + player["bench"]:
            if card is not None:
                visible += [card] + card.get("energyCards", []) + card.get("tools", [])
                visible += card.get("preEvolution", [])
        visible += [card for card in player["prize"] if card is not None]
        water_ids = {
            card_id
            for card_id, data in self.cards.items()
            if data["cardType"] == 5 and data["energyType"] == 3
        }
        energy = sum(card_id in water_ids for card_id in self.deck)
        energy -= sum(card["id"] in water_ids for card in visible)
        total = player["deckCount"] + sum(card is None for card in player["prize"])
        return max(0, min(energy, total)), total

    def attack_score(self, attack_id: int, current: Current) -> float:
        attack = self.attacks[attack_id]
        player = self.own(current)
        active = self.active(player)
        opponent = current["players"][1 - current["yourIndex"]]
        defending = self.active(opponent)
        damage = float(attack["damage"])
        penalty = 0.0
        water, unseen = self.unseen_water(current)
        if attack["name"] == "Hammer-lanche":
            draws = min(6, player["deckCount"])
            damage = 100 * draws * water / max(1, unseen)
            penalty = 25 if player["deckCount"] <= 12 else 0
        elif attack["name"] == "Riptide":
            damage = 20.0 * self.water_discard(current)
            penalty = 15
        elif attack["name"] == "Swirling Waves":
            penalty = 40
        elif attack["name"] == "Syrup Storm":
            damage += 30 * sum(
                card.get("energies", []).count(1)
                for card in player["active"] + player["bench"]
                if card is not None
            )
        elif attack["name"] == "Myriad Leaf Shower":
            damage += 30 * sum(
                len(card.get("energies", [])) for card in (active, defending) if card
            )
        elif attack["name"] == "Rapid-Fire Combo":
            damage += 50
        elif attack["name"] == "Trifrost":
            damage = 110
            penalty -= 110 * min(2, len(opponent["bench"]))
        elif attack["name"] == "Thunder Raid":
            damage = 0
            penalty -= max(
                (
                    min(210, card.get("hp", 0)) + 300 * (card.get("hp", 0) <= 210)
                    for card in opponent["bench"]
                    if card and self.cards[card["id"]]["ex"]
                ),
                default=0,
            )
        elif attack["name"] == "Seek Inspiration":
            damage = 150 if player["deckCount"] else 0
        elif attack["name"] == "Phantom Dive":
            penalty -= 60 if opponent["bench"] else 0
        elif attack["name"] == "Itchy Pollen":
            penalty -= 100 if current["turn"] < 6 else 30
        multiplier = 1
        resistance = 0
        if active and defending:
            attacking_type = self.cards[active["id"]]["energyType"]
            target = self.cards[defending["id"]]
            multiplier = 2 if target["weakness"] == attacking_type else 1
            resistance = 30 if target["resistance"] == attacking_type else 0
        effective = max(0, damage * multiplier - resistance)
        hp = defending.get("hp", 0) if defending else 0
        ko_probability = float(hp > 0 and effective >= hp)
        if attack["name"] == "Rapid-Fire Combo" and hp > 0:
            heads = max(0, ceil((hp + resistance - 200 * multiplier) / (50 * multiplier)))
            ko_probability = 0.5**heads
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
            energies = len(target.get("energies", []))
            name = self.cards[target["id"]]["name"]
            if data["cardType"] in (5, 6) and name not in ("Snover", "Mega Abomasnow ex", "Kyogre"):
                deficit = self.energy_deficit(target)
                gain = deficit - self.energy_deficit(target, data["energyType"])
                return (
                    400 * gain
                    + (80 if option.get("inPlayArea") == 4 and gain else 0)
                    + (30 if name == "Hydrapple ex" else 0)
                    - 30 * energies
                )
            if data["cardType"] == 5:
                desired = 1 if name == "Kyogre" and self.water_discard(current) >= 5 else 3
                if energies >= desired:
                    return -120
                score = 480 if energies < 2 and name != "Kyogre" else 70
                if name == "Kyogre":
                    score = 320 if energies == 0 and self.water_discard(current) >= 5 else 20
                return score + (80 if option.get("inPlayArea") == 4 else 0) - energies * 10
            return 340 if option.get("inPlayArea") == 4 else 50
        if kind == 7 and card is not None:
            name = self.cards[card["id"]]["name"]
            if name == "Lillie's Determination":
                return 360 if player["handCount"] <= 5 else -120
            if name in ("Ultra Ball", "Mega Signal"):
                return self.value(card, current) * 3
            if name == "Secret Box":
                return 190 if player["handCount"] >= 6 else -120
            if name in ("Surfing Beach", "Switch"):
                return (
                    160 if self.best_bench(current) > self.readiness(active, current) + 40 else -120
                )
            if self.cards[card["id"]]["basic"]:
                return (
                    self.value(card, current) * 3
                    if len(player["bench"]) < player["benchMax"]
                    else -120
                )
            if name == "Crispin":
                return 450 if active and self.energy_deficit(active) else 100
            if name == "Academy at Night":
                return 200
            if name == "Ciphermaniac’s Codebreaking":
                return 260
            if name == "Xerosic’s Machinations":
                hand = current["players"][1 - current["yourIndex"]]["handCount"]
                return 90 * (hand - 3) if hand > 3 else -120
            if name == "Boss’s Orders":
                opponent = current["players"][1 - current["yourIndex"]]
                targets = [
                    self.gust_value(target, current) for target in opponent["bench"] if target
                ]
                improvement = max(targets, default=0) - self.gust_value(
                    self.active(opponent), current
                )
                return 400 + improvement if improvement > 100 else -120
            return self.value(card, current) * 3
        if kind == 10 and card is not None:
            skills = {skill["name"] for skill in self.cards[card["id"]]["skills"]}
            if "Adrena-Brain" in skills:
                damaged = any(
                    entry and entry.get("maxHp", 0) > entry.get("hp", 0)
                    for entry in player["active"] + player["bench"]
                )
                return 700 if damaged else -120
            if skills & {"Teal Dance", "Ripening Charge"}:
                return 650 if any(c["id"] == 1 for c in player["hand"] or []) else -120
            if skills & {"Recon Directive", "Run Errand", "Flip the Script"}:
                return 620 if player["deckCount"] > 3 else -120
            if "Academy at Night" in skills:
                return 170 if active and self.cards[active["id"]]["name"] == "Slowking" else -120
            return -120
        if kind == 12:
            improvement = self.best_bench(current) - self.readiness(active, current)
            return 350 + improvement if improvement > 40 else -120
        if kind == 11:
            return -150
        if kind == 0:
            return float(option.get("number", 0)) if context in (38, 39, 40) else 0
        if kind in (1, 2):
            return float(kind == 1)
        if kind in (3, 4, 5, 6):
            if context == 26 and self.player_index(option, current) != current["yourIndex"]:
                host_option: Option = {**option, "type": 3}
                host = self.resolve(host_option, selection, current)
                if host:
                    energies = len(host.get("energies", []))
                    return 200 / max(1, energies) + (100 if option.get("area") == 4 else 0)
            if (
                context == 9
                and selection["effect"]
                and self.cards[selection["effect"]["id"]]["name"] == "Academy at Night"
            ):
                return self.copy_attack_value(card, current)
            if (
                context == 7
                and selection["effect"]
                and self.cards[selection["effect"]["id"]]["name"] == "Ciphermaniac’s Codebreaking"
            ):
                return self.copy_attack_value(card, current)
            if context in (8, 9, 10, 11, 26, 27, 29, 30, 31, 32):
                return 200 - self.value(card, current)
            if context == 1:
                if card and self.cards[card["id"]]["name"] == "Snover":
                    return 200
                if 121 in self.deck_cards and card:
                    return {235: 220, 119: 200, 112: 20}.get(card["id"], 50)
                return self.readiness(card, current) + (
                    100 if card and self.cards[card["id"]]["megaEx"] else 0
                )
            if context in (3, 4):
                if self.player_index(option, current) != current["yourIndex"]:
                    return self.gust_value(card, current)
                return self.readiness(card, current)
            if context in (13, 14, 15):
                if context == 13 and card:
                    hp = card.get("hp", 0)
                    return (1000 * self.prizes(card) if 0 < hp <= 30 else 0) - hp
                return 400 - (card.get("hp", 0) if card else 0)
            if context in (16, 17):
                return float(card.get("maxHp", 0) - card.get("hp", 0)) if card else 0
            if context == 22:
                return self.value(card, current)
            if context in (18, 21):
                if context == 21 and card:
                    return 300 - 50 * len(card.get("energies", [])) + 50 * self.energy_deficit(card)
                return self.readiness(card, current)
            return self.value(card, current)
        return 0

    def copy_attack_value(self, card: Card | None, current: Current) -> float:
        if card is None:
            return -100
        data = self.cards[card["id"]]
        if data["cardType"] != 0 or data["ex"] or data["megaEx"]:
            return -100
        return max((self.attack_score(i, current) for i in data["attacks"]), default=-100)

    def gust_value(self, card: Card | None, current: Current) -> float:
        if card is None:
            return 0
        active = self.active(self.own(current))
        damage = 200 if active and active["id"] == 121 and not self.energy_deficit(active) else 0
        hp = card.get("hp", 0)
        knockout = 1000 * self.prizes(card) if 0 < hp <= damage else 0
        return knockout + 400 - hp

    def best_bench(self, current: Current) -> float:
        return max(
            (self.readiness(card, current) for card in self.own(current)["bench"]), default=0
        )
