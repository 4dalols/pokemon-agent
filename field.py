"""Ladder panel: the most common 60-card list per archetype in public Playground replays.

Source: 1,310 public Playground replays (2026-09-29); ``entries`` is how many deck entries
played the archetype and ``win_rate`` its observed win rate. Slugs double as ``decks.DECKS`` names.
"""

from typing import TypedDict


class Archetype(TypedDict):
    name: str
    entries: int
    win_rate: float
    cards: dict[int, int]


PANEL: dict[str, Archetype] = {
    "dragapult_ex-fezandipiti_ex": {
        "name": "Dragapult ex / Fezandipiti ex",
        "entries": 150,
        "win_rate": 0.46,
        "cards": {
            2: 4,  # Basic {R} Energy
            5: 4,  # Basic {P} Energy
            7: 2,  # Basic {D} Energy
            112: 2,  # Munkidori
            119: 4,  # Dreepy
            120: 4,  # Drakloak
            121: 3,  # Dragapult ex
            140: 1,  # Fezandipiti ex
            235: 2,  # Budew
            1071: 1,  # Meowth ex
            1080: 1,  # Unfair Stamp
            1086: 4,  # Buddy-Buddy Poffin
            1097: 2,  # Night Stretcher
            1120: 4,  # Crushing Hammer
            1121: 4,  # Ultra Ball
            1152: 4,  # Poké Pad
            1182: 3,  # Boss’s Orders
            1198: 3,  # Crispin
            1213: 1,  # Judge
            1227: 4,  # Lillie's Determination
            1231: 1,  # Dawn
            1256: 2,  # Team Rocket's Watchtower
        },
    },
    "hydrapple_ex-teal_mask_ogerpon_ex": {
        "name": "Hydrapple ex / Teal Mask Ogerpon ex",
        "entries": 248,
        "win_rate": 0.565,
        "cards": {
            1: 14,  # Basic {G} Energy
            92: 2,  # Applin
            93: 2,  # Dipplin
            96: 4,  # Teal Mask Ogerpon ex
            140: 1,  # Fezandipiti ex
            150: 2,  # Hydrapple ex
            709: 2,  # Bayleef
            710: 2,  # Meganium
            917: 2,  # Chikorita
            920: 1,  # Tapu Bulu
            1071: 2,  # Meowth ex
            1080: 1,  # Unfair Stamp
            1094: 4,  # Bug Catching Set
            1097: 1,  # Night Stretcher
            1121: 4,  # Ultra Ball
            1152: 2,  # Poké Pad
            1182: 2,  # Boss’s Orders
            1184: 1,  # Lana’s Aid
            1213: 1,  # Judge
            1227: 4,  # Lillie's Determination
            1231: 2,  # Dawn
            1261: 4,  # Forest of Vitality
        },
    },
    "fezandipiti_ex-alakazam": {
        "name": "Fezandipiti ex / Alakazam",
        "entries": 118,
        "win_rate": 0.5,
        "cards": {
            5: 2,  # Basic {P} Energy
            13: 1,  # Enriching Energy
            19: 4,  # Telepath Psychic Energy
            66: 2,  # Dudunsparce
            140: 1,  # Fezandipiti ex
            305: 3,  # Dunsparce
            343: 1,  # Shaymin
            741: 4,  # Abra
            742: 4,  # Kadabra
            743: 4,  # Alakazam
            1079: 3,  # Rare Candy
            1081: 4,  # Enhanced Hammer
            1086: 4,  # Buddy-Buddy Poffin
            1097: 1,  # Night Stretcher
            1129: 1,  # Sacred Ash
            1152: 4,  # Poké Pad
            1182: 3,  # Boss’s Orders
            1184: 1,  # Lana’s Aid
            1197: 3,  # Xerosic’s Machinations
            1225: 4,  # Hilda
            1231: 4,  # Dawn
            1266: 2,  # Nighttime Mine
        },
    },
    "mega_abomasnow_ex-kyogre": {
        "name": "Mega Abomasnow ex / Kyogre",
        "entries": 244,
        "win_rate": 0.398,
        "cards": {
            3: 35,  # Basic {W} Energy
            721: 2,  # Kyogre
            722: 4,  # Snover
            723: 4,  # Mega Abomasnow ex
            1145: 4,  # Mega Signal
            1158: 1,  # Maximum Belt
            1205: 2,  # Cyrano
            1227: 4,  # Lillie's Determination
            1235: 4,  # Waitress
        },
    },
    "mega_lucario_ex-hariyama": {
        "name": "Mega Lucario ex / Hariyama",
        "entries": 103,
        "win_rate": 0.466,
        "cards": {
            6: 13,  # Basic {F} Energy
            673: 2,  # Makuhita
            674: 2,  # Hariyama
            675: 2,  # Lunatone
            676: 3,  # Solrock
            677: 3,  # Riolu
            678: 4,  # Mega Lucario ex
            1102: 4,  # Dusk Ball
            1123: 2,  # Switch
            1141: 4,  # Premium Power Pro
            1142: 4,  # Fighting Gong
            1152: 4,  # Poké Pad
            1159: 1,  # Hero’s Cape
            1182: 2,  # Boss’s Orders
            1192: 4,  # Carmine
            1227: 4,  # Lillie's Determination
            1252: 2,  # Gravity Mountain
        },
    },
    "mega_lopunny_ex-mega_froslass_ex": {
        "name": "Mega Lopunny ex / Mega Froslass ex",
        "entries": 170,
        "win_rate": 0.553,
        "cards": {
            3: 3,  # Basic {W} Energy
            11: 4,  # Mist Energy
            13: 1,  # Enriching Energy
            66: 3,  # Dudunsparce
            174: 1,  # Fan Rotom
            305: 4,  # Dunsparce
            848: 2,  # Buneary
            849: 2,  # Mega Lopunny ex
            860: 2,  # Snorunt
            861: 2,  # Mega Froslass ex
            1086: 4,  # Buddy-Buddy Poffin
            1087: 3,  # Hand Trimmer
            1121: 4,  # Ultra Ball
            1122: 2,  # Pokégear 3.0
            1152: 4,  # Poké Pad
            1174: 3,  # Air Balloon
            1182: 2,  # Boss’s Orders
            1225: 3,  # Hilda
            1227: 4,  # Lillie's Determination
            1229: 4,  # Wally's Compassion
            1264: 3,  # Battle Cage
        },
    },
    "dragapult_ex-munkidori": {
        "name": "Dragapult ex / Munkidori",
        "entries": 127,
        "win_rate": 0.425,
        "cards": {
            2: 7,  # Basic {R} Energy
            5: 6,  # Basic {P} Energy
            7: 2,  # Basic {D} Energy
            112: 2,  # Munkidori
            119: 4,  # Dreepy
            120: 4,  # Drakloak
            121: 3,  # Dragapult ex
            235: 3,  # Budew
            1080: 1,  # Unfair Stamp
            1086: 4,  # Buddy-Buddy Poffin
            1097: 2,  # Night Stretcher
            1120: 2,  # Crushing Hammer
            1121: 4,  # Ultra Ball
            1152: 4,  # Poké Pad
            1182: 3,  # Boss’s Orders
            1198: 2,  # Crispin
            1227: 4,  # Lillie's Determination
            1231: 2,  # Dawn
            1246: 1,  # Jamming Tower
        },
    },
    "marnies_grimmsnarl_ex-munkidori": {
        "name": "Marnie's Grimmsnarl ex / Munkidori",
        "entries": 118,
        "win_rate": 0.525,
        "cards": {
            7: 10,  # Basic {D} Energy
            104: 2,  # Froslass
            112: 4,  # Munkidori
            646: 4,  # Marnie's Impidimp
            647: 3,  # Marnie's Morgrem
            648: 3,  # Marnie's Grimmsnarl ex
            860: 2,  # Snorunt
            1079: 3,  # Rare Candy
            1080: 1,  # Unfair Stamp
            1086: 4,  # Buddy-Buddy Poffin
            1097: 3,  # Night Stretcher
            1122: 1,  # Pokégear 3.0
            1137: 1,  # Tool Scrapper
            1152: 4,  # Poké Pad
            1182: 2,  # Boss’s Orders
            1219: 4,  # Team Rocket's Petrel
            1227: 4,  # Lillie's Determination
            1231: 1,  # Dawn
            1259: 4,  # Spikemuth Gym
        },
    },
    "mega_kangaskhan_ex-latias_ex": {
        "name": "Mega Kangaskhan ex / Latias ex",
        "entries": 108,
        "win_rate": 0.565,
        "cards": {
            5: 4,  # Basic {P} Energy
            9: 1,  # Boomerang Energy
            19: 4,  # Telepath Psychic Energy
            140: 1,  # Fezandipiti ex
            144: 2,  # Kyurem
            162: 4,  # Slowpoke
            163: 3,  # Slowking
            183: 1,  # Smoochum
            184: 2,  # Latias ex
            224: 1,  # Annihilape
            756: 4,  # Mega Kangaskhan ex
            1071: 1,  # Meowth ex
            1088: 1,  # Prime Catcher
            1097: 2,  # Night Stretcher
            1121: 4,  # Ultra Ball
            1146: 2,  # Wondrous Patch
            1152: 4,  # Poké Pad
            1188: 4,  # Ciphermaniac’s Codebreaking
            1194: 3,  # Colress’s Tenacity
            1225: 2,  # Hilda
            1227: 4,  # Lillie's Determination
            1248: 4,  # Academy at Night
            1331: 2,  # Metagross
        },
    },
    "fezandipiti_ex-dusknoir": {
        "name": "Fezandipiti ex / Dusknoir",
        "entries": 104,
        "win_rate": 0.327,
        "cards": {
            5: 1,  # Basic {P} Energy
            19: 4,  # Telepath Psychic Energy
            109: 1,  # Abra
            131: 4,  # Duskull
            132: 2,  # Dusclops
            133: 2,  # Dusknoir
            140: 1,  # Fezandipiti ex
            235: 2,  # Budew
            343: 1,  # Shaymin
            741: 3,  # Abra
            742: 4,  # Kadabra
            743: 3,  # Alakazam
            1079: 4,  # Rare Candy
            1086: 3,  # Buddy-Buddy Poffin
            1088: 1,  # Prime Catcher
            1097: 2,  # Night Stretcher
            1121: 3,  # Ultra Ball
            1122: 2,  # Pokégear 3.0
            1129: 1,  # Sacred Ash
            1144: 3,  # Strange Timepiece
            1152: 4,  # Poké Pad
            1182: 1,  # Boss’s Orders
            1225: 4,  # Hilda
            1231: 4,  # Dawn
        },
    },
    "mega_lopunny_ex-dudunsparce_ex": {
        "name": "Mega Lopunny ex / Dudunsparce ex",
        "entries": 73,
        "win_rate": 0.562,
        "cards": {
            11: 4,  # Mist Energy
            13: 1,  # Enriching Energy
            14: 3,  # Spiky Energy
            65: 2,  # Dunsparce
            66: 3,  # Dudunsparce
            109: 1,  # Abra
            174: 1,  # Fan Rotom
            305: 2,  # Dunsparce
            306: 1,  # Dudunsparce ex
            848: 3,  # Buneary
            849: 3,  # Mega Lopunny ex
            1086: 4,  # Buddy-Buddy Poffin
            1121: 4,  # Ultra Ball
            1122: 4,  # Pokégear 3.0
            1152: 4,  # Poké Pad
            1174: 2,  # Air Balloon
            1182: 3,  # Boss’s Orders
            1225: 4,  # Hilda
            1227: 4,  # Lillie's Determination
            1229: 4,  # Wally's Compassion
            1264: 3,  # Battle Cage
        },
    },
    "mega_kangaskhan_ex-cornerstone_mask_ogerpon_ex": {
        "name": "Mega Kangaskhan ex / Cornerstone Mask Ogerpon ex",
        "entries": 59,
        "win_rate": 0.576,
        "cards": {
            1: 1,  # Basic {G} Energy
            11: 4,  # Mist Energy
            14: 4,  # Spiky Energy
            18: 4,  # Grow Grass Energy
            20: 2,  # Rock Fighting Energy
            117: 1,  # Cornerstone Mask Ogerpon ex
            344: 4,  # Dwebble
            345: 4,  # Crustle
            756: 2,  # Mega Kangaskhan ex
            1086: 2,  # Buddy-Buddy Poffin
            1112: 1,  # Super Potion
            1121: 2,  # Ultra Ball
            1122: 4,  # Pokégear 3.0
            1123: 1,  # Switch
            1147: 4,  # Jumbo Ice Cream
            1159: 1,  # Hero’s Cape
            1182: 4,  # Boss’s Orders
            1194: 2,  # Colress’s Tenacity
            1197: 1,  # Xerosic’s Machinations
            1219: 4,  # Team Rocket's Petrel
            1225: 2,  # Hilda
            1227: 4,  # Lillie's Determination
            1257: 1,  # Team Rocket's Factory
            1261: 1,  # Forest of Vitality
        },
    },
    "mega_kangaskhan_ex-crustle-shaymin": {
        "name": "Mega Kangaskhan ex / Crustle / Shaymin",
        "entries": 45,
        "win_rate": 0.667,
        "cards": {
            1: 1,  # Basic {G} Energy
            11: 4,  # Mist Energy
            14: 4,  # Spiky Energy
            18: 4,  # Grow Grass Energy
            343: 1,  # Shaymin
            344: 4,  # Dwebble
            345: 4,  # Crustle
            756: 4,  # Mega Kangaskhan ex
            1086: 4,  # Buddy-Buddy Poffin
            1087: 1,  # Hand Trimmer
            1122: 4,  # Pokégear 3.0
            1123: 4,  # Switch
            1147: 4,  # Jumbo Ice Cream
            1159: 1,  # Hero’s Cape
            1182: 2,  # Boss’s Orders
            1197: 4,  # Xerosic’s Machinations
            1225: 4,  # Hilda
            1227: 4,  # Lillie's Determination
            1264: 2,  # Battle Cage
        },
    },
    "thwackey-dipplin": {
        "name": "Thwackey / Dipplin",
        "entries": 54,
        "win_rate": 0.519,
        "cards": {
            1: 8,  # Basic {G} Energy
            42: 3,  # Applin
            88: 3,  # Volbeat
            89: 4,  # Grookey
            90: 4,  # Thwackey
            92: 1,  # Applin
            93: 4,  # Dipplin
            343: 1,  # Shaymin
            1080: 1,  # Unfair Stamp
            1086: 4,  # Buddy-Buddy Poffin
            1094: 4,  # Bug Catching Set
            1097: 2,  # Night Stretcher
            1129: 1,  # Sacred Ash
            1152: 4,  # Poké Pad
            1175: 1,  # Brave Bangle
            1182: 1,  # Boss’s Orders
            1210: 1,  # Brock’s Scouting
            1211: 1,  # Black Belt’s Training
            1225: 4,  # Hilda
            1227: 4,  # Lillie's Determination
            1245: 4,  # Festival Grounds
        },
    },
    "mega_lucario_ex-throh": {
        "name": "Mega Lucario ex / Throh",
        "entries": 53,
        "win_rate": 0.415,
        "cards": {
            6: 22,  # Basic {F} Energy
            531: 2,  # Throh
            677: 4,  # Riolu
            678: 4,  # Mega Lucario ex
            1082: 1,  # Hyper Aroma
            1097: 2,  # Night Stretcher
            1118: 2,  # Energy Retrieval
            1121: 4,  # Ultra Ball
            1122: 4,  # Pokégear 3.0
            1123: 2,  # Switch
            1182: 3,  # Boss’s Orders
            1185: 3,  # Explorer’s Guidance
            1189: 4,  # Salvatore
            1192: 3,  # Carmine
        },
    },
}


def panel_weights() -> dict[str, float]:
    """Share of ladder entries per archetype (sums to 1)."""
    total = sum(archetype["entries"] for archetype in PANEL.values())
    return {slug: archetype["entries"] / total for slug, archetype in PANEL.items()}
