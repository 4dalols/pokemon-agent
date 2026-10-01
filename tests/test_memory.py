import copy
import random
from collections import Counter
from typing import cast

from archetypes import LIBRARY, Predictor
from benchmark import opponent_action
from decks import deck_list
from engine import Battle, battle_finish, battle_select, battle_start
from main import ATTACKS, CARDS, POLICY
from memory import DECK, HAND, PRIZE, Threat, Tracker
from policy import Policy
from schema import Card, Current, LogEntry, Observation, Player, Selection


def player(deck_count: int = 40, hand_count: int = 1) -> Player:
    return {
        "deckCount": deck_count,
        "handCount": hand_count,
        "hand": None,
        "discard": [],
        "active": [cast(Card, {"id": 722, "serial": 5, "hp": 70, "maxHp": 70})],
        "bench": [],
        "benchMax": 5,
        "prize": [None] * 6,
    }


def observation(turn: int = 2, round_: int = 1, logs: list[LogEntry] | None = None) -> Observation:
    current: Current = {
        "players": [player(), player()],
        "yourIndex": 0,
        "turn": turn,
        "turnActionCount": 0,
        "result": -1,
        "stadium": [],
        "looking": None,
        "energyAttached": False,
        "supporterPlayed": False,
    }
    current["round"] = round_
    return {"select": None, "current": current, "logs": logs}


def test_tracker_resets_on_new_round_and_survives_malformed_logs() -> None:
    tracker = Tracker(POLICY.deck)
    tracker.observe(observation(logs=[{"type": 10, "playerIndex": 1, "cardId": 121, "serial": 70}]))
    assert tracker.sides[1].identity == {70: 121}
    garbage = cast(list[LogEntry], [None, 3, {"type": "x"}, {"type": 6, "playerIndex": 7}])
    tracker.observe(observation(turn=3, logs=garbage))
    assert tracker.sides[1].identity == {70: 121}
    tracker.observe(observation(turn=1, round_=2))
    assert tracker.sides[1].identity == {} and tracker.games == 2
    tracker.observe({"select": None, "current": None})
    tracker.observe(cast(Observation, {"select": None, "current": observation()["current"]}))


def test_hidden_moves_spread_known_identities_across_zones() -> None:
    tracker = Tracker(POLICY.deck)
    logs: list[LogEntry] = [
        {
            "type": 6,
            "playerIndex": 1,
            "cardId": 1121,
            "serial": 61,
            "fromArea": DECK,
            "toArea": HAND,
        },
        {"type": 7, "playerIndex": 1, "fromArea": HAND, "toArea": DECK},
        {"type": 7, "playerIndex": 1, "fromArea": HAND, "toArea": DECK},
    ]
    tracker.observe(observation(logs=logs))
    side = tracker.sides[1]
    assert side.hidden[61][DECK] == 1.0 and side.hidden[61][HAND] == 0.0
    tracker.observe(
        observation(turn=3, logs=[{"type": 7, "playerIndex": 1, "fromArea": DECK, "toArea": PRIZE}])
    )
    assert 0 < side.hidden[61][PRIZE] < 0.1
    assert abs(sum(side.hidden[61].values()) - 1.0) < 1e-9
    current = observation()["current"]
    assert current is not None
    assert tracker.revealed(current["players"][1], current) == Counter({1121: 1, 722: 1})


def test_predictor_identifies_dragapult_and_samples_legal_zone_sizes() -> None:
    predictor = Predictor(deck_list("abomasnow"), CARDS, random.Random(3))
    revealed = Counter({119: 2, 120: 1, 121: 1, 1086: 1})
    posterior = predictor.posterior(revealed)
    assert max(posterior, key=posterior.__getitem__) == "dragapult-ex-top"
    abomasnow = predictor.posterior(Counter({721: 1, 723: 1}))
    assert sum(w for n, w in abomasnow.items() if "abomasnow" in n or n == "mirror") > 0.95
    tracker = Tracker(POLICY.deck)
    tracker.observe(observation(logs=[{"type": 4, "playerIndex": 1, "cardId": 1086, "serial": 90}]))
    current = observation()["current"]
    assert current is not None
    theirs = current["players"][1]
    sampled = predictor.sample(theirs, current, tracker.sides[1], revealed)
    assert sampled is not None
    deck, hand, prizes = sampled
    assert (len(deck), len(hand), len(prizes)) == (40, 1, 6)
    assert hand == [1086]
    assert all(card_id in CARDS for card_id in deck + prizes)
    assert predictor.posterior(Counter({1: 20, 2: 20, 3: 20})) == {}


def test_threat_detects_bench_snipers_and_limits_bench_development() -> None:
    tracker = Tracker(POLICY.deck)
    logs: list[LogEntry] = [
        {"type": 10, "playerIndex": 1, "cardId": 121, "serial": 70},
        {"type": 16, "playerIndex": 0, "cardId": 722, "serial": 5, "value": -60},
    ]
    tracker.observe(observation(logs=logs))
    current = observation()["current"]
    assert current is not None
    threat = tracker.threat(current, CARDS, ATTACKS)
    assert threat.bench_sniper and threat.max_damage == 60
    assert Tracker(POLICY.deck).threat(current, CARDS, ATTACKS) == Threat()
    policy = Policy(POLICY.deck, CARDS, ATTACKS)
    selection: Selection = {
        "type": 0,
        "context": 0,
        "minCount": 1,
        "maxCount": 1,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "deck": None,
        "contextCard": None,
        "effect": None,
        "option": [{"type": 7, "area": 2, "index": 0, "playerIndex": 0}],
    }
    board = copy.deepcopy(current)
    board["players"][0]["hand"] = [cast(Card, {"id": 722, "serial": 9})]
    board["players"][0]["bench"] = [
        cast(Card, {"id": 721, "serial": 1, "hp": 150, "maxHp": 150})
    ] * 2
    assert policy.score(selection["option"][0], selection, board) > 0
    policy.threat = threat
    assert policy.score(selection["option"][0], selection, board) == -120
    snover = cast(Card, {"id": 722, "serial": 5, "hp": 60, "maxHp": 70})
    assert policy.readiness(snover, board) < Policy(POLICY.deck, CARDS, ATTACKS).readiness(
        snover, board
    )


def test_full_deck_search_pins_down_own_prizes_in_a_native_game() -> None:
    tracker = Tracker(POLICY.deck)
    rng = random.Random(5)
    known = False
    try:
        for _ in range(6):
            obs = cast(Observation, battle_start(POLICY.deck, POLICY.deck)[0])
            while True:
                current, selection = obs["current"], obs["select"]
                assert current is not None and selection is not None
                if current["result"] >= 0:
                    break
                if current["yourIndex"] == 0:
                    tracker.observe(obs)
                    mine = current["players"][0]
                    exact = tracker.own_hidden(mine, current)
                    if exact is not None:
                        deck, prizes = exact
                        assert len(deck) == mine["deckCount"]
                        assert len(prizes) == sum(card is None for card in mine["prize"])
                        assert Counter(deck + prizes) <= Counter(POLICY.deck)
                        known = True
                    action = POLICY.choose(obs)
                else:
                    action = opponent_action(obs, "greedy", rng)
                obs = cast(Observation, battle_select(action))
            battle_finish()
            Battle.battle_ptr = None
            if known:
                break
    finally:
        battle_finish()
        Battle.battle_ptr = None
    assert known
    assert all(sum(entry.cards.values()) == 60 for entry in LIBRARY)


def test_threat_lists_the_pokemon_the_opponent_has_shown() -> None:
    tracker = Tracker(POLICY.deck)
    logs: list[LogEntry] = [
        {"type": 10, "playerIndex": 1, "cardId": 678, "serial": 70},
        {"type": 10, "playerIndex": 1, "cardId": 1182, "serial": 71},
    ]
    tracker.observe(observation(logs=logs))
    current = observation()["current"]
    assert current is not None
    assert tracker.threat(current, CARDS, ATTACKS).attackers == (678,)
