"""
Human-readable descriptive tilesets for different games, keyed by tile char to a plain-English
description. These drive two things for the LLM captioning pipeline:
  - the tile-set key shown to the model in the prompt, and the object-count names in the
    deterministic structural metadata ("tiles" below)
  - game-specific prompt plug-ins folded into the captioning system prompt: freeform
    vocabulary/tone guidance ("prompt_vocab") and extra rule paragraphs unique to this game's
    quirks ("prompt_rules"), such as MM2's multi-tile-object dedup instruction

They are separate from the tileset JSONs referenced by common_settings.py (MM.json, MMLV.json,
mm2_tileset_we.json, ...), which carry the structural descriptors (solid/passable/enemy/null/...)
and the id<->char maps used for grid analysis. A tile char here must exactly match a key in that
game's tileset JSON, or it will silently be skipped by filter_tile_set()/deterministic_caption().

To add a game: define its <GAME>_TILESET_DICT below and register it in GAMES at the bottom.
"""

from util.common_settings import MM_FULL_TILESET, MMLV_TILESET, MM_SIMPLE_TILESET, MM2_TILESET


MM_FULL_TILESET_DICT = {
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
    "tiles": {
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

# Simplified Mega Man tileset: matches MM-Simple scene encodings. All enemies are collapsed
# into a single enemy tile, and most power-up/metadata markers are grouped with the existing
# power-up tile.
MM_SIMPLE_TILESET_DICT = {
    "tiles": {
        "#": "Solid blocks representing ground or walls",
        "-": "Air",
        "@": "Out of bounds, inaccessible null space",
        "A": "Disappearing/Reappearing blocks (fades in and out)",
        "B": "Solid but breakable blocks",
        "C": "Hazard Blocks: extends a temporary passable but damaging hazard outward",
        "D": "Passable Door blocks",
        "H": "Deadly solid hazard",
        "M": "Moving Platform blocks",
        "a": "Enemy",
        "l": "Power-up or collectible",
        "|": "Climbable ladders",
        "~": "Water (slows movement)",
    }
}

# Mario Maker 2 tileset: all 67 tiles (MM2_TILE_COUNT in common_settings.py), derived from
# mm2_tileset_we.json's structural tags. Note MM2's tileset has no "spawn" or "null" tagged
# tile -- unlike MM/MMLV, it doesn't encode player spawn or out-of-bounds space as a grid tile.
MM2_TILESET_DICT = {
    "tiles": {
        " ": "Air",
        "#": "Solid blocks representing ground or walls",
        "B": "Solid but breakable brick block",
        "?": "Solid question block (yields a hidden item when hit)",
        "c": "Collectible coin",
        "g": "Ground-walking, damaging Goomba enemy",
        "K": "Ground-walking, damaging Koopa enemy",
        "P": "Damaging Piranha Plant enemy",
        "t": "Moving, damaging Thwomp enemy",
        "^": "Solid, damaging spike hazard",
        "N": "Solid note block (bounces the player/emits a note when hit)",
        "T": "Solid mushroom platform",
        "=": "Solid bridge platform",
        "k": "Passable semisolid platform (can be jumped up through from below)",
        "i": "Collectible Fire Flower power-up",
        "V": "Solid, damaging Bullet Bill Blaster (shooter enemy)",
        "|": "Solid pipe/warp",

        "H": "Solid hard block",
        "h": "Passable hidden block (reveals an item when hit from below)",
        "d": "Passable donut block (falls shortly after the player stands on it)",
        "I": "Solid, slippery ice block",
        "O": "Solid on/off block (toggles solid state when a nearby switch is hit)",
        ".": "Passable dotted-line block (toggles between solid and passable)",

        "D": "Passable door/warp",
        "f": "Passable mid-level checkpoint flag",
        "G": "Passable goal flagpole (level exit)",

        "m": "Moving, damaging Hammer Bro enemy",
        "o": "Explosive, damaging Bob-omb enemy",
        "s": "Moving, damaging Spiny enemy",
        "b": "Moving, damaging Buzzy Beetle enemy",
        "L": "Flying, damaging Lakitu enemy",
        "Z": "Moving, damaging Banzai Bill projectile enemy",
        "y": "Flying, damaging Magikoopa enemy",
        "u": "Moving, damaging Boo enemy",
        "X": "Moving, damaging Bowser boss enemy",
        "x": "Moving, damaging Bowser Jr. boss enemy",
        "@": "Damaging Chain Chomp enemy",
        "~": "Moving, damaging Cheep Cheep enemy",
        "q": "Moving, damaging Blooper enemy",
        "w": "Moving, damaging Wiggler enemy",
        "&": "Moving, damaging Lava Bubble enemy",
        "r": "Moving, damaging Rocky Wrench enemy",
        "n": "Moving, damaging Monty Mole enemy",
        "!": "Moving, damaging Boom Boom boss enemy",
        "9": "Moving, damaging Dry Bones enemy",
        "A": "Moving, damaging Angry Sun enemy",

        "U": "Collectible 1-Up power-up",
        "*": "Collectible Super Star power-up",
        "M": "Collectible Super Mushroom power-up",
        "E": "Collectible game-style-specific power-up",
        "S": "Collectible P Switch",
        "W": "Collectible POW block",
        "J": "Collectible spring",
        "z": "Moving, damaging rideable enemy/vehicle (Goomba's Shoe, Yoshi's Egg, or Yoshi, depending on game style)",

        "-": "Passable moving lift platform",
        "F": "Solid moving lava lift platform",
        "j": "Passable, interactive swinging claw",

        "e": "Damaging rotating fire bar hazard",
        "%": "Moving, damaging saw hazard",
        "l": "Damaging burner hazard (periodic flame jet)",
        "0": "Solid, moving, damaging skewer hazard",
        "8": "Passable, moving, damaging twister hazard",

        "<": "Solid cloud platform block",
        "[": "Passable, climbable vine",
        "]": "Passable one-way platform",
        ";": "Passable Clown Car vehicle",
        ")": "Solid cannon (shoots projectiles)",
    }
}

# MM2's one unique extra RULE paragraph for the captioning prompt: many of its object types
# occupy more than one grid cell per placed instance (pipes, platforms, blasters, etc.), so
# without this the LLM tends to count one object per cell instead of per placed instance.
# Other games don't need this and should leave GAMES[...]["prompt_rules"] as [].
MM2_MULTI_TILE_RULE = (
    "MULTI-TILE OBJECTS: Many object types occupy more than one grid cell per placed "
    "instance, appearing as a contiguous block of identical cells (e.g. pipes, bullet "
    "bill blasters, bridges, platforms). Treat one contiguous block of the same tile "
    "type as ONE object, not one object per cell -- unless the block's shape clearly "
    "looks like several same-size chunks repeated side by side, in which case count each "
    "repeated chunk as its own object. If two blocks of the same tile type are not "
    "touching, they are separate objects."
)

# Registry of captionable games, keyed by the --game CLI value. Each entry carries:
#   name:         human-readable game name injected into the LLM captioning prompt for context
#   tiles:        the descriptive tileset dict above (char -> readable description) used for
#                 the prompt's tile-set key and the object-count names in the deterministic
#                 metadata
#   tileset:      tileset JSON (structural descriptors + id<->char maps) matching that game's
#                 encoded scenes, from common_settings.py
#   prompt_vocab: list of freeform vocabulary/tone/naming guidance strings specific to this
#                 game, spliced into the captioning system prompt. [] if this game has none
#                 (the default -- fill in later as desired, it's optional polish, not required)
#   prompt_rules: list of extra RULE-like paragraphs specific to this game's quirks (e.g.
#                 MM2's multi-tile-object dedup instruction above). [] if this game needs none
GAMES = {
    "MM-Simple": {
        "name": "Mega Man (Simple)",
        "tiles": MM_SIMPLE_TILESET_DICT,
        "tileset": MM_SIMPLE_TILESET,
        "prompt_vocab": [],
        "prompt_rules": [],
    },
    "MM-Full": {
        "name": "Mega Man",
        "tiles": MM_FULL_TILESET_DICT,
        "tileset": MM_FULL_TILESET,
        "prompt_vocab": [],
        "prompt_rules": [],
    },
    "MMLV": {
        "name": "Mega Man Maker",
        "tiles": MMLV_TILESET_DICT,
        "tileset": MMLV_TILESET,
        "prompt_vocab": [],
        "prompt_rules": [],
    },
    "MM2": {
        "name": "Mario Maker 2",
        "tiles": MM2_TILESET_DICT,
        "tileset": MM2_TILESET,
        # TODO: fill in MM2-specific vocabulary/tone guidance (e.g. how to refer to enemies,
        # editor terminology) once you have a feel for what the captions need.
        "prompt_vocab": [],
        "prompt_rules": [MM2_MULTI_TILE_RULE],
    },
}