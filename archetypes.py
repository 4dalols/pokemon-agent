"""Opponent deck library and the hidden-card predictor built on it.

Each archetype is a complete 60-card list (R2 card ids) reconstructed from the public
Playground replays. The predictor scores every list by how many of the opponent's
revealed cards it fails to explain, mixes that with the list's ladder frequency and
samples the opponent's hidden cards (deck, hand, prizes) from the chosen list.
"""

import itertools
import math
import random
from collections import Counter
from dataclasses import dataclass

from memory import DECK, HAND, PRIZE, Side, visible
from schema import CardData, Current, Player

UNEXPLAINED_PENALTY = 2.5
MIN_TOTAL_WEIGHT = 1e-4


@dataclass(frozen=True)
class Archetype:
    name: str
    prior: float
    cards: Counter[int]


def archetype(name: str, prior: float, counts: dict[int, int]) -> Archetype:
    cards = Counter(counts)
    if sum(cards.values()) != 60:
        raise ValueError(f"{name}: archetype lists must hold exactly 60 cards")
    return Archetype(name, prior, cards)


LIBRARY: tuple[Archetype, ...] = (
    # Dragapult ex: exact list of the top-2 Playground teams (ratings 1018 / 964)
    archetype(
        "dragapult-ex-top",
        0.1000,
        {
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
    ),
    # Hydrapple exx2 / Teal Mask Ogerpon exx4 / Fezandipiti exx1 / Meowth exx2
    archetype(
        "hydrapple-ex-1",
        0.0947,
        {
            1: 14,
            92: 2,
            93: 2,
            96: 4,
            140: 1,
            150: 2,
            709: 2,
            710: 2,
            917: 2,
            920: 1,
            1071: 2,
            1080: 1,
            1094: 4,
            1097: 1,
            1121: 4,
            1152: 2,
            1182: 2,
            1184: 1,
            1213: 1,
            1227: 4,
            1231: 2,
            1261: 4,
        },
    ),
    # Mega Abomasnow exx4 / Kyogrex2 / Snoverx4
    archetype(
        "mega-abomasnow-ex-2",
        0.0931,
        {3: 35, 721: 2, 722: 4, 723: 4, 1145: 4, 1158: 1, 1205: 2, 1227: 4, 1235: 4},
    ),
    # Mega Lopunny exx2 / Mega Froslass exx2 / Dudunsparcex3 / Fan Rotomx1
    archetype(
        "mega-lopunny-ex-3",
        0.0649,
        {
            3: 3,
            11: 4,
            13: 1,
            66: 3,
            174: 1,
            305: 4,
            848: 2,
            849: 2,
            860: 2,
            861: 2,
            1086: 4,
            1087: 3,
            1121: 4,
            1122: 2,
            1152: 4,
            1174: 3,
            1182: 2,
            1225: 3,
            1227: 4,
            1229: 4,
            1264: 3,
        },
    ),
    # Dragapult exx3 / Fezandipiti exx1 / Meowth exx1 / Munkidorix2
    archetype(
        "dragapult-ex-4",
        0.0573,
        {
            2: 4,
            5: 4,
            7: 2,
            112: 2,
            119: 4,
            120: 4,
            121: 3,
            140: 1,
            235: 2,
            1071: 1,
            1080: 1,
            1086: 4,
            1097: 2,
            1120: 4,
            1121: 4,
            1152: 4,
            1182: 3,
            1198: 3,
            1213: 1,
            1227: 4,
            1231: 1,
            1256: 2,
        },
    ),
    # Dragapult exx3 / Fezandipiti exx1 / Latias exx1 / Meowth exx1
    archetype(
        "dragapult-ex-5",
        0.0489,
        {
            2: 4,
            5: 4,
            119: 4,
            120: 4,
            121: 3,
            140: 1,
            184: 1,
            235: 2,
            1071: 1,
            1079: 2,
            1080: 1,
            1086: 4,
            1097: 2,
            1120: 4,
            1121: 4,
            1152: 3,
            1156: 1,
            1182: 3,
            1198: 4,
            1210: 2,
            1227: 4,
            1256: 2,
        },
    ),
    # Dragapult exx3 / Munkidorix2 / Drakloakx4 / Dreepyx4
    archetype(
        "dragapult-ex-6",
        0.0485,
        {
            2: 7,
            5: 6,
            7: 2,
            112: 2,
            119: 4,
            120: 4,
            121: 3,
            235: 3,
            1080: 1,
            1086: 4,
            1097: 2,
            1120: 2,
            1121: 4,
            1152: 4,
            1182: 3,
            1198: 2,
            1227: 4,
            1231: 2,
            1246: 1,
        },
    ),
    # Marnie's Grimmsnarl exx3 / Munkidorix4 / Marnie's Morgremx3 / Froslassx2
    archetype(
        "marnie-s-grimmsnarl-ex-7",
        0.0450,
        {
            7: 10,
            104: 2,
            112: 4,
            646: 4,
            647: 3,
            648: 3,
            860: 2,
            1079: 3,
            1080: 1,
            1086: 4,
            1097: 3,
            1122: 1,
            1137: 1,
            1152: 4,
            1182: 2,
            1219: 4,
            1227: 4,
            1231: 1,
            1259: 4,
        },
    ),
    # Fezandipiti exx1 / Dudunsparcex2 / Alakazamx4 / Kadabrax4
    archetype(
        "fezandipiti-ex-8",
        0.0450,
        {
            5: 2,
            13: 1,
            19: 4,
            66: 2,
            140: 1,
            305: 3,
            343: 1,
            741: 4,
            742: 4,
            743: 4,
            1079: 3,
            1081: 4,
            1086: 4,
            1097: 1,
            1129: 1,
            1152: 4,
            1182: 3,
            1184: 1,
            1197: 3,
            1225: 4,
            1231: 4,
            1266: 2,
        },
    ),
    # Mega Kangaskhan exx4 / Fezandipiti exx1 / Latias exx2 / Metagrossx2
    archetype(
        "mega-kangaskhan-ex-9",
        0.0412,
        {
            5: 4,
            9: 1,
            19: 4,
            140: 1,
            144: 2,
            162: 4,
            163: 3,
            183: 1,
            184: 2,
            224: 1,
            756: 4,
            1071: 1,
            1088: 1,
            1097: 2,
            1121: 4,
            1146: 2,
            1152: 4,
            1188: 4,
            1194: 3,
            1225: 2,
            1227: 4,
            1248: 4,
            1331: 2,
        },
    ),
    # Fezandipiti exx1 / Dusknoirx2 / Alakazamx3 / Dusclopsx2
    archetype(
        "fezandipiti-ex-10",
        0.0397,
        {
            5: 1,
            19: 4,
            109: 1,
            131: 4,
            132: 2,
            133: 2,
            140: 1,
            235: 2,
            343: 1,
            741: 3,
            742: 4,
            743: 3,
            1079: 4,
            1086: 3,
            1088: 1,
            1097: 2,
            1121: 3,
            1122: 2,
            1129: 1,
            1144: 3,
            1152: 4,
            1182: 1,
            1225: 4,
            1231: 4,
        },
    ),
    # Mega Lucario exx4 / Hariyamax2 / Lunatonex2 / Solrockx3
    archetype(
        "mega-lucario-ex-11",
        0.0393,
        {
            6: 13,
            673: 2,
            674: 2,
            675: 2,
            676: 3,
            677: 3,
            678: 4,
            1102: 4,
            1123: 2,
            1141: 4,
            1142: 4,
            1152: 4,
            1159: 1,
            1182: 2,
            1192: 4,
            1227: 4,
            1252: 2,
        },
    ),
    # Dragapult exx3 / Fezandipiti exx1 / Meowth exx1 / Dudunsparcex1
    archetype(
        "dragapult-ex-12",
        0.0374,
        {
            2: 3,
            5: 3,
            7: 3,
            66: 1,
            112: 2,
            119: 4,
            120: 4,
            121: 3,
            140: 1,
            235: 2,
            305: 1,
            1071: 1,
            1080: 1,
            1086: 4,
            1097: 3,
            1120: 4,
            1121: 3,
            1152: 4,
            1182: 3,
            1198: 2,
            1213: 1,
            1227: 4,
            1240: 1,
            1260: 2,
        },
    ),
    # Fezandipiti exx1 / Dudunsparcex2 / Alakazamx4 / Kadabrax4
    archetype(
        "fezandipiti-ex-13",
        0.0286,
        {
            5: 2,
            13: 1,
            19: 4,
            66: 2,
            140: 1,
            305: 3,
            343: 1,
            741: 4,
            742: 4,
            743: 4,
            1079: 4,
            1081: 4,
            1086: 4,
            1097: 1,
            1129: 1,
            1152: 4,
            1182: 2,
            1184: 1,
            1197: 2,
            1225: 4,
            1231: 4,
            1264: 3,
        },
    ),
    # Mega Lopunny exx3 / Dudunsparce exx1 / Dudunsparcex3 / Fan Rotomx1
    archetype(
        "mega-lopunny-ex-14",
        0.0279,
        {
            11: 4,
            13: 1,
            14: 3,
            65: 2,
            66: 3,
            109: 1,
            174: 1,
            305: 2,
            306: 1,
            848: 3,
            849: 3,
            1086: 4,
            1121: 4,
            1122: 4,
            1152: 4,
            1174: 2,
            1182: 3,
            1225: 4,
            1227: 4,
            1229: 4,
            1264: 3,
        },
    ),
    # Dragapult exx3 / Fezandipiti exx1 / Meowth exx1 / Munkidorix2
    archetype(
        "dragapult-ex-15",
        0.0240,
        {
            2: 4,
            5: 4,
            7: 2,
            112: 2,
            119: 4,
            120: 4,
            121: 3,
            140: 1,
            235: 2,
            1071: 1,
            1080: 1,
            1086: 4,
            1097: 2,
            1120: 4,
            1121: 4,
            1152: 4,
            1182: 3,
            1198: 3,
            1213: 1,
            1227: 4,
            1231: 1,
            1246: 2,
        },
    ),
    # Mega Kangaskhan exx2 / Cornerstone Mask Ogerpon exx1 / Crustlex4 / Dwebblex4
    archetype(
        "mega-kangaskhan-ex-16",
        0.0225,
        {
            1: 1,
            11: 4,
            14: 4,
            18: 4,
            20: 2,
            117: 1,
            344: 4,
            345: 4,
            756: 2,
            1086: 2,
            1112: 1,
            1121: 2,
            1122: 4,
            1123: 1,
            1147: 4,
            1159: 1,
            1182: 4,
            1194: 2,
            1197: 1,
            1219: 4,
            1225: 2,
            1227: 4,
            1257: 1,
            1261: 1,
        },
    ),
    # Fezandipiti exx1 / Dudunsparcex3 / Alakazamx4 / Genesectx1
    archetype(
        "fezandipiti-ex-17",
        0.0218,
        {
            5: 2,
            13: 1,
            19: 4,
            65: 3,
            66: 3,
            140: 1,
            142: 1,
            343: 1,
            741: 4,
            742: 4,
            743: 4,
            1079: 3,
            1081: 4,
            1086: 4,
            1097: 1,
            1129: 1,
            1152: 4,
            1156: 1,
            1182: 2,
            1184: 1,
            1225: 3,
            1231: 4,
            1264: 4,
        },
    ),
    # Hydrapple exx2 / Teal Mask Ogerpon exx4 / Fezandipiti exx1 / Meowth exx2
    archetype(
        "hydrapple-ex-18",
        0.0206,
        {
            1: 14,
            93: 2,
            96: 4,
            140: 1,
            149: 2,
            150: 2,
            708: 2,
            709: 2,
            710: 2,
            920: 1,
            1071: 2,
            1080: 1,
            1094: 4,
            1097: 1,
            1121: 4,
            1152: 2,
            1182: 2,
            1184: 1,
            1213: 1,
            1227: 4,
            1231: 2,
            1261: 4,
        },
    ),
    # Thwackeyx4 / Shayminx1 / Dipplinx4 / Volbeatx3
    archetype(
        "thwackey-19",
        0.0206,
        {
            1: 8,
            42: 3,
            88: 3,
            89: 4,
            90: 4,
            92: 1,
            93: 4,
            343: 1,
            1080: 1,
            1086: 4,
            1094: 4,
            1097: 2,
            1129: 1,
            1152: 4,
            1175: 1,
            1182: 1,
            1210: 1,
            1211: 1,
            1225: 4,
            1227: 4,
            1245: 4,
        },
    ),
    # Mega Lucario exx4 / Throhx2 / Riolux4
    archetype(
        "mega-lucario-ex-20",
        0.0202,
        {
            6: 22,
            531: 2,
            677: 4,
            678: 4,
            1082: 1,
            1097: 2,
            1118: 2,
            1121: 4,
            1122: 4,
            1123: 2,
            1182: 3,
            1185: 3,
            1189: 4,
            1192: 3,
        },
    ),
    # Fezandipiti exx1 / Dudunsparcex2 / Alakazamx3 / Genesectx1
    archetype(
        "fezandipiti-ex-21",
        0.0198,
        {
            5: 2,
            13: 1,
            19: 4,
            66: 2,
            140: 1,
            142: 1,
            305: 3,
            343: 1,
            741: 4,
            742: 4,
            743: 3,
            858: 1,
            1079: 3,
            1081: 3,
            1086: 4,
            1097: 1,
            1129: 1,
            1152: 4,
            1156: 3,
            1182: 2,
            1225: 4,
            1231: 4,
            1264: 4,
        },
    ),
    # Mega Lucario exx4 / Hariyamax2 / Lunatonex2 / Solrockx3
    archetype(
        "mega-lucario-ex-23",
        0.0187,
        {
            6: 14,
            673: 2,
            674: 2,
            675: 2,
            676: 3,
            677: 4,
            678: 4,
            1102: 4,
            1123: 2,
            1141: 4,
            1142: 4,
            1152: 2,
            1159: 1,
            1182: 3,
            1192: 4,
            1227: 4,
            1252: 1,
        },
    ),
    # Marnie's Grimmsnarl exx4 / Fezandipiti exx1 / Munkidorix2 / Marnie's Morgremx3
    archetype(
        "marnie-s-grimmsnarl-ex-24",
        0.0179,
        {
            7: 15,
            112: 2,
            140: 1,
            646: 4,
            647: 3,
            648: 4,
            649: 2,
            1079: 4,
            1086: 4,
            1097: 2,
            1121: 4,
            1123: 2,
            1126: 1,
            1152: 3,
            1182: 2,
            1227: 4,
            1231: 3,
        },
    ),
    # Mega Lucario exx4 / Hariyamax2 / Lunatonex2 / Solrockx3
    archetype(
        "mega-lucario-ex-25",
        0.0179,
        {
            6: 13,
            673: 2,
            674: 2,
            675: 2,
            676: 3,
            677: 3,
            678: 4,
            1121: 4,
            1123: 2,
            1141: 4,
            1142: 4,
            1152: 4,
            1159: 1,
            1182: 2,
            1213: 4,
            1227: 4,
            1229: 2,
        },
    ),
    # Archaludon exx4 / Cinderacex4 / Duraludonx4 / Relicanthx1
    archetype(
        "archaludon-ex-26",
        0.0172,
        {
            8: 11,
            57: 1,
            169: 4,
            190: 4,
            666: 4,
            1097: 3,
            1121: 4,
            1122: 4,
            1147: 4,
            1152: 4,
            1159: 1,
            1182: 3,
            1185: 4,
            1213: 1,
            1227: 4,
            1244: 4,
        },
    ),
    # Mega Abomasnow exx4 / Kyogrex2 / Snoverx4
    archetype(
        "mega-abomasnow-ex-27",
        0.0145,
        {
            3: 24,
            721: 2,
            722: 4,
            723: 4,
            1097: 2,
            1121: 2,
            1123: 1,
            1126: 1,
            1145: 4,
            1152: 4,
            1182: 3,
            1205: 2,
            1227: 4,
            1235: 2,
            1262: 1,
        },
    ),
    # Dudunsparcex3 / Alakazamx4 / Kadabrax4 / Dunsparcex4
    archetype(
        "dudunsparce-28",
        0.0092,
        {
            5: 3,
            13: 1,
            19: 4,
            66: 3,
            305: 4,
            741: 4,
            742: 4,
            743: 4,
            1079: 4,
            1081: 4,
            1086: 4,
            1097: 3,
            1129: 1,
            1152: 4,
            1182: 3,
            1184: 1,
            1225: 4,
            1231: 4,
            1264: 1,
        },
    ),
    # Mega Excadrill exx2 / Genesect exx2 / Fezandipiti exx1 / Metagrossx2
    archetype(
        "mega-excadrill-ex-29",
        0.0061,
        {
            8: 16,
            85: 4,
            86: 4,
            140: 1,
            547: 2,
            1086: 2,
            1121: 1,
            1122: 2,
            1126: 1,
            1134: 4,
            1139: 1,
            1147: 1,
            1182: 3,
            1191: 1,
            1210: 1,
            1213: 1,
            1219: 4,
            1227: 2,
            1252: 1,
            1331: 2,
            1387: 3,
            1406: 2,
            1413: 1,
        },
    ),
    # Mega Lucario exx3 / Hariyamax2 / Crustlex2 / Lunatonex2
    archetype(
        "mega-lucario-ex-30",
        0.0046,
        {
            6: 16,
            344: 2,
            345: 2,
            673: 2,
            674: 2,
            675: 2,
            676: 2,
            677: 4,
            678: 3,
            1086: 4,
            1097: 2,
            1107: 1,
            1118: 2,
            1119: 1,
            1121: 4,
            1123: 2,
            1141: 2,
            1182: 2,
            1227: 3,
            1231: 2,
        },
    ),
    # Reshiram exx4 / Gouging Fire exx4
    archetype(
        "reshiram-ex-31",
        0.0027,
        {
            2: 10,
            3: 7,
            46: 4,
            573: 4,
            1097: 2,
            1102: 2,
            1107: 1,
            1116: 2,
            1118: 3,
            1121: 4,
            1122: 4,
            1123: 3,
            1182: 3,
            1185: 3,
            1192: 2,
            1198: 4,
            1207: 2,
        },
    ),
    # Mega Kangaskhan exx2 / Fezandipiti exx1 / Cornerstone Mask Ogerpon exx1 / Crustlex4
    archetype(
        "mega-kangaskhan-ex-32",
        0.0027,
        {
            7: 1,
            9: 1,
            11: 4,
            14: 4,
            18: 4,
            20: 2,
            112: 1,
            117: 1,
            140: 1,
            344: 4,
            345: 4,
            756: 2,
            1086: 2,
            1112: 1,
            1121: 2,
            1122: 4,
            1123: 1,
            1147: 3,
            1159: 1,
            1182: 4,
            1194: 2,
            1219: 4,
            1225: 2,
            1227: 4,
            1261: 1,
        },
    ),
    # LugiaEXx4 / Zangoose exx4
    archetype(
        "lugiaex-33",
        0.0023,
        {
            3: 14,
            337: 4,
            1002: 4,
            1097: 2,
            1102: 2,
            1116: 2,
            1118: 2,
            1121: 4,
            1122: 3,
            1123: 4,
            1182: 3,
            1185: 4,
            1192: 2,
            1203: 3,
            1205: 3,
            1207: 4,
        },
    ),
    # Marnie's Grimmsnarl exx3 / Munkidorix4 / Marnie's Morgremx3 / Froslassx2
    archetype(
        "marnie-s-grimmsnarl-ex-34",
        0.0023,
        {
            7: 9,
            11: 1,
            104: 2,
            112: 4,
            646: 4,
            647: 3,
            648: 3,
            860: 2,
            1079: 4,
            1080: 1,
            1086: 4,
            1097: 3,
            1121: 1,
            1122: 1,
            1152: 4,
            1182: 2,
            1219: 2,
            1227: 4,
            1231: 2,
            1259: 4,
        },
    ),
    # Mega Zeraora exx4 / Fezandipiti exx2 / Iono’s Kilowattrelx2 / Miraidonx4
    archetype(
        "mega-zeraora-ex-35",
        0.0023,
        {
            4: 12,
            140: 2,
            270: 2,
            271: 2,
            1038: 2,
            1080: 1,
            1086: 4,
            1097: 2,
            1121: 4,
            1123: 2,
            1152: 4,
            1182: 2,
            1192: 2,
            1227: 4,
            1231: 1,
            1252: 2,
            1271: 4,
            1369: 4,
            1370: 4,
        },
    ),
    # Archaludon exx4 / Fezandipiti exx2 / Duraludonx4 / Munkidorix1
    archetype(
        "archaludon-ex-36",
        0.0019,
        {
            2: 1,
            7: 1,
            8: 10,
            16: 1,
            57: 3,
            112: 1,
            140: 2,
            169: 4,
            190: 4,
            1097: 3,
            1121: 4,
            1122: 4,
            1147: 2,
            1152: 4,
            1159: 1,
            1182: 3,
            1185: 4,
            1227: 4,
            1244: 4,
        },
    ),
)


class Sampler:
    """Cheap per-rollout determinizations of the opponent's hidden zones."""

    def __init__(
        self,
        rng: random.Random,
        pools: list[Counter[int]],
        weights: list[float],
        sizes: dict[int, int],
        placements: list[tuple[int, list[int], list[float]]],
        filler: int,
    ) -> None:
        self.rng = rng
        self.pools = pools
        self.cumulative = list(itertools.accumulate(weights))
        self.sizes = sizes
        self.placements = placements
        self.filler = filler

    def sample(self) -> tuple[list[int], list[int], list[int]]:
        pool = self.pools[self.rng.choices(range(len(self.pools)), cum_weights=self.cumulative)[0]]
        pool = pool.copy()
        zones: dict[int, list[int]] = {DECK: [], HAND: [], PRIZE: []}
        for card_id, areas, probabilities in self.placements:
            if pool[card_id] <= 0:
                continue
            area = self.rng.choices(areas, probabilities)[0]
            if len(zones[area]) < self.sizes[area]:
                zones[area].append(card_id)
                pool[card_id] -= 1
        rest = list(pool.elements())
        self.rng.shuffle(rest)
        needed = sum(self.sizes.values()) - sum(len(zone) for zone in zones.values())
        if len(rest) < needed:
            rest.extend([self.filler] * (needed - len(rest)))
        for area, zone in zones.items():
            take = self.sizes[area] - len(zone)
            zone.extend(rest[:take])
            del rest[:take]
            self.rng.shuffle(zone)
        return zones[DECK], zones[HAND], zones[PRIZE]


class Predictor:
    def __init__(
        self,
        own_deck: list[int],
        cards: dict[int, CardData],
        rng: random.Random,
        library: tuple[Archetype, ...] = LIBRARY,
    ) -> None:
        mirror = archetype("mirror", 0.1, dict(Counter(own_deck)))
        self.library = (*library, mirror)
        self.cards = cards
        self.rng = rng
        self.basic_energy = {
            data["energyType"]: card_id
            for card_id, data in cards.items()
            if data["cardType"] == 5 and data["name"].startswith("Basic ")
        }

    def weights(self, revealed: Counter[int]) -> list[float]:
        weights = []
        for entry in self.library:
            unexplained = sum((revealed - entry.cards).values())
            weights.append(entry.prior * math.exp(-UNEXPLAINED_PENALTY * unexplained))
        return weights

    def posterior(self, revealed: Counter[int]) -> dict[str, float]:
        weights = self.weights(revealed)
        total = sum(weights)
        if total < MIN_TOTAL_WEIGHT:
            return {}
        return {e.name: w / total for e, w in zip(self.library, weights, strict=True)}

    def best(self, revealed: Counter[int]) -> Archetype | None:
        weights = self.weights(revealed)
        if sum(weights) < MIN_TOTAL_WEIGHT:
            return None
        return self.library[max(range(len(weights)), key=weights.__getitem__)]

    def choose(self, revealed: Counter[int]) -> Archetype | None:
        weights = self.weights(revealed)
        if sum(weights) < MIN_TOTAL_WEIGHT:
            return None
        return self.rng.choices(self.library, weights)[0]

    def sample(
        self, theirs: Player, current: Current, side: Side, revealed: Counter[int]
    ) -> tuple[list[int], list[int], list[int]] | None:
        """Sample (deck, hand, prizes) for the opponent, or None if no list fits."""
        sampler = self.prepare(theirs, current, side, revealed)
        return None if sampler is None else sampler.sample()

    def prepare(
        self, theirs: Player, current: Current, side: Side, revealed: Counter[int]
    ) -> Sampler | None:
        """Precompute everything that is constant for one decision's determinizations."""
        weights = self.weights(revealed)
        if sum(weights) < MIN_TOTAL_WEIGHT:
            return None
        seen = Counter(visible(theirs, current))
        sizes = {
            DECK: theirs["deckCount"],
            HAND: theirs["handCount"],
            PRIZE: sum(card is None for card in theirs["prize"]),
        }
        known = [
            (card_id, [a for a in sizes if probabilities[a] > 0])
            for serial, probabilities in side.hidden.items()
            if (card_id := side.identity.get(serial)) is not None
        ]
        placements = [
            (card_id, areas, [side.hidden[serial][a] for a in areas])
            for (card_id, areas), serial in zip(known, side.hidden, strict=True)
            if areas
        ]
        pools = [entry.cards - seen for entry in self.library]
        return Sampler(self.rng, pools, weights, sizes, placements, self.filler(theirs))

    def filler(self, theirs: Player) -> int:
        types = Counter(
            self.cards[card["id"]]["energyType"]
            for card in theirs["active"] + theirs["bench"]
            if card is not None and card["id"] in self.cards
        )
        energy_type = types.most_common(1)[0][0] if types else 11
        return self.basic_energy.get(energy_type, self.basic_energy.get(11, 3))
