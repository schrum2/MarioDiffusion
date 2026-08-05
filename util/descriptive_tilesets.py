"""
Human-readable descriptive tilesets per Mega Man game, keyed by tile char to a plain-English
description. These drive the LLM captioning prompt (the tile-set key shown to the model) and the
object-count names in the deterministic structural metadata. They are separate from the tileset
JSONs under datasets/ (MM.json, MMLV.json, ...), which carry the structural descriptors
(solid/passable/enemy/null) and the id<->char maps used for grid analysis.

To add a game: define its <GAME>_TILESET_DICT below and register it in GAMES at the bottom.
"""

MM_FULL_TILESET_DICT = {
    "tiles" : {
        "P": "Mega Man's starting spawn point",
        "Z": "Level exit point/final goal",
        "@": "Out of bounds, inaccessible null space",
        "-": "Air",
        "~": "Water (slows movement)",
        "#": "Solid blocks representing ground or walls",
        "|": "Climbable ladders",
        "B": "Solid but breakable blocks",
        "L": "Large Life Energy power-up",  
        "H": "Deadly solid hazard",
        "q": "Jumping Kamadoma enemy",
        "o": "Flying Mambu enemy",
        "j": "Flying Bunby Heli enemy",
        "c": "Ranged wall-mounted Blaster enemy",
        "^": "Vertical Adhering Suzy enemy",
        "<": "Horizontal Adhering Suzy enemy",
        "f": "Jumping Big Eye enemy",
        "t": "Secret transparent blocks (looks like regular blocks, but Mega Man can phase through them)",
        "A": "Disappearing/Reappearing blocks (fades in and out)",
        "M": "Moving Platform blocks",
        "D": "Passable Door blocks",
        "W": "Large Weapon Energy power-up",
        "w": "Small Weapon Energy power-up",
        "l": "Small Life Energy power-up",
        "+": "Collectible 1-UP Extra Life Power-up",
        "*": "Collectible Yashichi Power-up",
        "U": "Collectible Magnet Beam Power-up",
        "C": "Hazard Blocks: extends a temporary passable but damaging hazard outward",
        "p": "Ranged Foot Holder enemy, Mega Man can stand and jump from these",
        "r": "Ranged, shielded Sniper Joe enemy",
        "k": "Killer Bomb enemy",
        "g": "Ground Gabyoall enemy",
        "e": "Stationary, ranged Screw Driver enemy",
        "m": "Jumping exploding Bombombomb enemy",
        "i": "Floating, ranged Watcher enemy",
        "b": "Flying Bunby Heli enemy (green)",
        "a": "Stationary, ranged Met enemy",
        "d": "Ranged Pickelman enemy",
        "h": "Crazy Razy enemy",
        "n": "Flying PePe penguin enemy",
        "I": "Tackle Fire Enemies",
    }
}

MMLV_TILESET_DICT = {
    "tiles" : {
        "-": "Air",
        "@": "Out of bounds, inaccessible null space",
        "F": "Falling platform: a solid block that drops when stood on",
        "~": "Water (slows movement)",
        "#": "Solid blocks representing ground or walls",
        "|": "Climbable ladders",
        "B": "Solid but breakable blocks",
        "t": "Fake blocks (look solid, but Mega Man can phase through them)",
        "A": "Disappearing/Reappearing blocks (fades in and out)",
        "M": "Moving Platform blocks",
        "D": "Passable Door blocks (boss/level doors)",
        "W": "Large Weapon Energy power-up",
        "w": "Small Weapon Energy power-up",
        "L": "Large Life Energy power-up",
        "l": "Small Life Energy power-up",
        "+": "Collectible 1-UP Extra Life power-up",
        "*": "Collectible Yashichi power-up (full restore)",
        "P": "Mega Man's starting spawn point",
        "Z": "Level exit point/final goal",
        ">": "Conveyor belt that pushes right",
        "E": "Conveyor belt that pushes left",
        "T": "Teleporter that warps Mega Man to a paired teleporter",
        "H": "Deadly solid spike hazard",
        "C": "Hazard Blocks: extends a temporary passable but damaging hazard outward",
        "x": "Fan that blows Mega Man upward",
        "!": "Deadly lava (instant death like spikes)",
        "p": "Ranged Foot Holder enemy, Mega Man can stand and jump from these",
        "r": "Ranged, shielded Sniper Joe enemy",
        "o": "Mambu (flying shell) enemy spawner",
        "n": "Mambu (flying shell) enemy",
        "k": "Killer Bullet enemy spawner",
        "j": "Flying Killer Bullet enemy",
        "g": "Ground Spine enemy",
        "c": "Ranged, wall-mounted Beak enemy",
        "e": "Stationary, ranged Screw Bomber enemy",
        "I": "Tackle Fire enemies",
        "i": "Floating, ranged Watcher enemy",
        "^": "Vertical Octopus Battery enemy",
        "<": "Horizontal Octopus Battery enemy",
        "f": "Jumping Big Eye enemy",
        "a": "Stationary, ranged Met enemy",
        "d": "Ranged Picket Man enemy",
        "h": "Crazy Razy enemy",
        "=": "Moving platform path/track (the rail a moving platform rides)",
        "s": "Spring that bounces Mega Man upward",
        "V": "Vertical key door, opened with a key (behaves like a breakable barrier)",
        "Y": "Horizontal key door, opened with a key (behaves like a breakable barrier)",
        "K": "Collectible key that opens key doors",
        "R": "Rising platform: a solid block that rises",
        "G": "Horizontal fire emitter shooting fire to the right (damaging hazard)",
        "J": "Horizontal fire emitter shooting fire to the left (damaging hazard)",
    }
}


from util.common_settings import MM_FULL_TILESET, MMLV_TILESET, MM_SIMPLE_TILESET

# Built-in simple Mega Man tileset: same scene encoding as the full game, but all enemies
# are described generically. This lets the captions focus on structural features while
# treating all combatants as one generic enemy type.
MM_SIMPLE_TILESET_DICT = {
    "tiles": {
        "P": "Mega Man's starting spawn point",
        "Z": "Level exit point/final goal",
        "@": "Out of bounds, inaccessible null space",
        "-": "Air",
        "~": "Water (slows movement)",
        "#": "Solid blocks representing ground or walls",
        "|": "Climbable ladders",
        "B": "Solid but breakable blocks",
        "L": "Large Life Energy power-up",
        "H": "Deadly solid hazard",
        "t": "Secret transparent blocks (looks like regular blocks, but Mega Man can phase through them)",
        "A": "Disappearing/Reappearing blocks (fades in and out)",
        "M": "Moving Platform blocks",
        "D": "Passable Door blocks",
        "W": "Large Weapon Energy power-up",
        "w": "Small Weapon Energy power-up",
        "l": "Small Life Energy power-up",
        "+": "Collectible 1-UP Extra Life Power-up",
        "*": "Collectible Yashichi Power-up",
        "U": "Collectible Magnet Beam Power-up",
        "C": "Hazard Blocks: extends a temporary passable but damaging hazard outward",
        "q": "Enemy",
        "o": "Enemy",
        "j": "Enemy",
        "c": "Enemy",
        "^": "Enemy",
        "<": "Enemy",
        "f": "Enemy",
        "p": "Enemy",
        "r": "Enemy",
        "k": "Enemy",
        "g": "Enemy",
        "e": "Enemy",
        "m": "Enemy",
        "i": "Enemy",
        "b": "Enemy",
        "a": "Enemy",
        "d": "Enemy",
        "h": "Enemy",
        "n": "Enemy",
        "I": "Enemy",
    }
}

# Registry of captionable Mega Man games, keyed by the --game CLI value. Each entry carries:
#   name:    human-readable game name injected into the LLM captioning prompt for context
#   tiles:   the descriptive tileset dict above (char -> readable description) used for the
#            prompt's tile-set key and the object-count names in the deterministic metadata
#   tileset: default tileset JSON (structural descriptors + id<->char maps) matching that
#            game's encoded scenes; overridable via --tileset
MM_FULL_GAME = {
    "name": "Mega Man",
    "tiles": MM_FULL_TILESET_DICT,
    "tileset": MM_FULL_TILESET,
}

GAMES = {
    "megaman": MM_FULL_GAME,
    "MM-Full": MM_FULL_GAME,
    "MM-Simple": {
        "name": "Mega Man (Simple)",
        "tiles": MM_SIMPLE_TILESET_DICT,
        "tileset": MM_SIMPLE_TILESET,
    },
    "MMLV": {
        "name": "Mega Man Maker",
        "tiles": MMLV_TILESET_DICT,
        "tileset": MMLV_TILESET,
    },
}
