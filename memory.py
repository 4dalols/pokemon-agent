"""Per-game memory built from the incremental event logs in each observation.

The engine observation is memoryless, but every observation carries the event logs
emitted since this player's previous observation. The tracker turns them into facts
the hidden-state predictor can use: the identity of opponent cards that were revealed
and then went back into a hidden zone (hand, deck or prizes), our own prize cards once
a full-deck search has shown the whole deck, the damage each side has dealt, coin
flips and energy attachments. Malformed or missing logs never raise; the tracker then
simply knows less.
"""

from collections import Counter

from schema import Card, Current, LogEntry, Observation, Player

DECK, HAND, DISCARD, ACTIVE, BENCH, PRIZE = 1, 2, 3, 4, 5, 6
LOG_DRAW, LOG_MOVE, LOG_MOVE_HIDDEN, LOG_PLAY, LOG_ATTACH = 4, 6, 7, 10, 11
LOG_EVOLVE, LOG_ATTACK, LOG_HP, LOG_COIN = 12, 15, 16, 22


class Side:
    """What is remembered about one player's cards.

    `identity` maps a physical card (its serial) to the card id once its face was
    seen. `hidden` holds, for cards currently out of sight, the probability of each
    hidden zone (deck, hand, prize); hidden moves spread that probability according
    to the zone sizes at the time of the move.
    """

    def __init__(self) -> None:
        self.identity: dict[int, int] = {}
        self.hidden: dict[int, dict[int, float]] = {}
        self.counts: dict[int, int] = {DECK: 60, HAND: 0, PRIZE: 0}
        self.attacks: list[int] = []
        self.damage_dealt: list[int] = []
        self.coins: list[bool] = []
        self.energy_attached: Counter[int] = Counter()

    def reveal(self, serial: int, card_id: int, area: int | None) -> None:
        self.identity[serial] = card_id
        if area in self.counts:
            self.hidden[serial] = {zone: float(zone == area) for zone in self.counts}
        else:
            self.hidden.pop(serial, None)

    def move(self, source: object, destination: object) -> None:
        """A card moved face down; every hidden card in `source` may be the one."""
        from_area = source if isinstance(source, int) else -1
        to_area = destination if isinstance(destination, int) else -1
        if from_area in self.counts and self.hidden:
            size = max(1, self.counts[from_area])
            for probabilities in self.hidden.values():
                share = probabilities[from_area] / size
                probabilities[from_area] -= share
                if to_area in self.counts:
                    probabilities[to_area] += share
        if from_area in self.counts:
            self.counts[from_area] = max(0, self.counts[from_area] - 1)
        if to_area in self.counts:
            self.counts[to_area] += 1
        if to_area not in self.counts and from_area in self.counts:
            self.drop_lost()

    def drop_lost(self) -> None:
        for serial in [s for s, p in self.hidden.items() if sum(p.values()) < 0.02]:
            del self.hidden[serial]

    def sync(self, player: Player) -> None:
        self.counts = {
            DECK: player["deckCount"],
            HAND: player["handCount"],
            PRIZE: sum(card is None for card in player["prize"]),
        }

    def expected(self, area: int) -> dict[int, float]:
        """Expected number of copies of each known card sitting in a hidden zone."""
        totals: dict[int, float] = {}
        for serial, probabilities in self.hidden.items():
            if serial in self.identity and probabilities[area] > 0:
                card_id = self.identity[serial]
                totals[card_id] = totals.get(card_id, 0.0) + probabilities[area]
        return totals


class Tracker:
    def __init__(self, deck: list[int]) -> None:
        self.deck_pool = Counter(deck)
        self.round: int | None = None
        self.turn = -1
        self.sides = (Side(), Side())
        self.own_prizes: Counter[int] | None = None
        self.in_play: Counter[int] = Counter()
        self.games = 0

    def reset(self) -> None:
        self.sides = (Side(), Side())
        self.own_prizes = None
        self.games += 1

    def observe(self, observation: Observation) -> None:
        current = observation["current"]
        if current is None:
            return
        round_ = current.get("round")
        if round_ != self.round or current["turn"] < self.turn:
            self.reset()
        self.round, self.turn = round_, current["turn"]
        self.ingest(observation.get("logs") or [])
        self.in_play = playing(observation, current, current["yourIndex"])
        for side, player in zip(self.sides, current["players"], strict=False):
            side.sync(player)
            for serial in visible_serials(player):
                side.hidden.pop(serial, None)
        self.update_prizes(observation, current)

    def ingest(self, logs: list[LogEntry]) -> None:
        for log in logs:
            if not isinstance(log, dict):
                continue
            kind, player = log.get("type"), log.get("playerIndex")
            if not isinstance(kind, int) or player not in (0, 1):
                continue
            side = self.sides[player]
            serial, card_id = log.get("serial"), log.get("cardId")
            if kind == LOG_DRAW and isinstance(serial, int) and isinstance(card_id, int):
                side.reveal(serial, card_id, HAND)
                side.move(DECK, HAND)
            elif kind == LOG_MOVE and isinstance(serial, int) and isinstance(card_id, int):
                from_area, to_area = log.get("fromArea"), log.get("toArea")
                side.reveal(serial, card_id, to_area if isinstance(to_area, int) else None)
                if isinstance(from_area, int) and from_area in side.counts:
                    side.counts[from_area] = max(0, side.counts[from_area] - 1)
                if isinstance(to_area, int) and to_area in side.counts:
                    side.counts[to_area] += 1
            elif kind == LOG_MOVE_HIDDEN:
                side.move(log.get("fromArea"), log.get("toArea"))
            elif kind in (LOG_PLAY, LOG_ATTACH, LOG_EVOLVE):
                if isinstance(serial, int) and isinstance(card_id, int):
                    side.reveal(serial, card_id, BENCH)
                    if kind == LOG_ATTACH:
                        side.energy_attached[card_id] += 1
            elif kind == LOG_ATTACK:
                attack = log.get("attackId")
                if isinstance(attack, int):
                    side.attacks.append(attack)
            elif kind == LOG_HP:
                value = log.get("value")
                if isinstance(value, int) and value < 0:
                    self.sides[1 - player].damage_dealt.append(-value)
            elif kind == LOG_COIN:
                head = log.get("head")
                if isinstance(head, bool):
                    side.coins.append(head)

    # Own prizes -----------------------------------------------------------------

    def update_prizes(self, observation: Observation, current: Current) -> None:
        mine = current["players"][current["yourIndex"]]
        hidden = sum(card is None for card in mine["prize"])
        if self.own_prizes is not None:
            if sum(self.own_prizes.values()) != hidden:
                self.own_prizes = None
            else:
                return
        selection = observation["select"]
        if selection is None or selection["deck"] is None:
            return
        if len(selection["deck"]) != mine["deckCount"]:
            return
        shown = Counter(card["id"] for card in selection["deck"]) + self.in_play
        remaining = self.deck_pool - Counter(visible(mine, current)) - shown
        if sum(remaining.values()) == hidden and all(n >= 0 for n in remaining.values()):
            self.own_prizes = remaining

    # Queries ----------------------------------------------------------------------

    def own_hidden(self, mine: Player, current: Current) -> tuple[list[int], list[int]] | None:
        """Exact (deck, prizes) multisets for our own hidden cards, when known."""
        if self.own_prizes is None:
            return None
        unseen = self.deck_pool - Counter(visible(mine, current)) - self.in_play
        deck = unseen - self.own_prizes
        if sum(deck.values()) != mine["deckCount"] or not self.own_prizes <= unseen:
            return None
        return list(deck.elements()), list(self.own_prizes.elements())

    def opponent(self, current: Current) -> Side:
        return self.sides[1 - current["yourIndex"]]

    def revealed(self, theirs: Player, current: Current) -> Counter[int]:
        """Every distinct opponent card whose identity has been seen this game."""
        side = self.opponent(current)
        return Counter(side.identity.values()) | Counter(visible(theirs, current))

    def max_damage(self, current: Current) -> int:
        return max(self.opponent(current).damage_dealt, default=0)


def visible(player: Player, current: Current) -> list[int]:
    ids = [card["id"] for card in player["hand"] or []]
    ids.extend(card["id"] for card in player["discard"])
    ids.extend(card["id"] for card in player["prize"] if card is not None)
    for card in player["active"] + player["bench"]:
        if card is not None:
            ids.extend(attached_ids(card))
    owner = current["players"].index(player)
    ids.extend(card["id"] for card in current["stadium"] if card["playerIndex"] == owner)
    return ids


def playing(observation: Observation, current: Current, owner: int) -> Counter[int]:
    """Own cards that are mid-resolution and therefore in no public zone."""
    selection = observation["select"]
    if selection is None:
        return Counter()
    shown = set(visible_serials(current["players"][owner]))
    shown.update(card["serial"] for card in current["stadium"])
    cards = {
        c["serial"]: c["id"]
        for c in (selection["effect"], selection["contextCard"])
        if c is not None and c["playerIndex"] == owner and c["serial"] not in shown
    }
    return Counter(cards.values())


def visible_serials(player: Player) -> list[int]:
    serials = [card["serial"] for card in player["hand"] or []]
    serials.extend(card["serial"] for card in player["discard"])
    for card in player["active"] + player["bench"]:
        if card is not None:
            for attached in [card] + card.get("energyCards", []) + card.get("tools", []):
                serials.append(attached["serial"])
            serials.extend(pre["serial"] for pre in card.get("preEvolution", []))
    return serials


def attached_ids(card: Card) -> list[int]:
    ids = [card["id"]]
    for attached in card.get("energyCards", []) + card.get("tools", []):
        ids.append(attached["id"])
    ids.extend(pre["id"] for pre in card.get("preEvolution", []))
    return ids
