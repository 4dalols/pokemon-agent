from typing import NotRequired, TypedDict


class Card(TypedDict):
    id: int
    serial: int
    playerIndex: int
    hp: NotRequired[int]
    maxHp: NotRequired[int]
    energies: NotRequired[list[int]]
    energyCards: NotRequired[list["Card"]]
    tools: NotRequired[list["Card"]]
    preEvolution: NotRequired[list["Card"]]


class Player(TypedDict):
    active: list[Card | None]
    bench: list[Card | None]
    hand: list[Card] | None
    discard: list[Card]
    prize: list[Card | None]
    deckCount: int
    handCount: int
    benchMax: int
    poisoned: NotRequired[bool]
    burned: NotRequired[bool]
    asleep: NotRequired[bool]
    paralyzed: NotRequired[bool]
    confused: NotRequired[bool]


class Current(TypedDict):
    players: list[Player]
    yourIndex: int
    turn: int
    turnActionCount: int
    result: int
    stadium: list[Card]
    looking: list[Card | None] | None
    energyAttached: bool
    supporterPlayed: bool
    firstPlayer: NotRequired[int]
    win: NotRequired[int]
    draw: NotRequired[int]
    round: NotRequired[int]


class Option(TypedDict):
    type: int
    index: NotRequired[int]
    area: NotRequired[int]
    playerIndex: NotRequired[int | None]
    inPlayArea: NotRequired[int]
    inPlayIndex: NotRequired[int]
    energyIndex: NotRequired[int]
    toolIndex: NotRequired[int]
    attackId: NotRequired[int]
    number: NotRequired[int]
    count: NotRequired[int]
    cardId: NotRequired[int]


class Selection(TypedDict):
    type: int
    context: int
    minCount: int
    maxCount: int
    remainDamageCounter: int
    remainEnergyCost: int
    option: list[Option]
    deck: list[Card] | None
    contextCard: Card | None
    effect: Card | None


class LogEntry(TypedDict, total=False):
    type: int
    playerIndex: int
    cardId: int
    serial: int
    cardIdTarget: int
    serialTarget: int
    fromArea: int
    toArea: int
    attackId: int
    value: int
    head: bool


class Observation(TypedDict):
    select: Selection | None
    current: Current | None
    logs: NotRequired[list[LogEntry] | None]
    search_begin_input: NotRequired[str | None]
    remainingOverageTime: NotRequired[float]


class Skill(TypedDict):
    name: str
    text: str


class CardData(TypedDict):
    cardId: int
    name: str
    cardType: int
    hp: int
    energyType: int
    weakness: int | None
    resistance: int | None
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    megaEx: bool
    aceSpec: bool
    evolvesFrom: str | None
    retreatCost: int
    attacks: list[int]
    skills: list[Skill]


class AttackData(TypedDict):
    attackId: int
    name: str
    text: str
    damage: int
    energies: list[int]


class Catalog(TypedDict):
    source: str
    cards: list[CardData]
    attacks: list[AttackData]
