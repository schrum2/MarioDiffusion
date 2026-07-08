"""Object metadata for the conversion pipeline, anchored on the canonical
training tileset ``mm2_tileset_we.json``.

The tileset defines the glyph vocabulary the diffusion model learns; the last
descriptor of each glyph is its canonical object name. ``OBJ_META`` maps each
decoder object name to the glyph used to draw it; ``ASCII_REPLACEMENTS`` and
``ASCII_DROP`` cover objects with no native glyph (3D World exclusives and other
out-of-tileset objects are filtered upstream during extraction); ``NAME_TO_ID``
and the gamestyle/theme maps drive the reverse (ASCII -> JSON) direction.

Id tables tied to a binary format live with their stage instead: ``OBJ_ID`` in
:mod:`mm2pipeline_data.bcd` (the level.ksy parser) and ``OBJ_ID_MAP`` in
:mod:`mm2pipeline_data.swe` (the SMMWE save format).
"""
import json

from . import paths

# ---------------------------------------------------------------------------
# Canonical tileset (mm2_tileset_we.json): glyph -> descriptor list, with the
# last descriptor naming the object. GLYPH_TO_NAME / NAME_TO_GLYPH expose that
# vocabulary for any consumer that needs to map between glyphs and names.
# ---------------------------------------------------------------------------

with open(paths.MM2_TILESET_PATH, encoding="utf-8") as _f:
    TILESET = json.load(_f)["tiles"]

GLYPH_TO_NAME = {glyph: descriptors[-1] for glyph, descriptors in TILESET.items()}
NAME_TO_GLYPH = {}
for _glyph, _name in GLYPH_TO_NAME.items():
    NAME_TO_GLYPH.setdefault(_name, _glyph)


# ---------------------------------------------------------------------------
# Object categories
# ---------------------------------------------------------------------------

CAT_TERRAIN  = "terrain"
CAT_ENEMY    = "enemy"
CAT_ITEM     = "item"
CAT_PLATFORM = "platform"
CAT_DOOR     = "door"
CAT_HAZARD   = "hazard"
CAT_DECO     = "deco"
CAT_OTHER    = "other"

# Decoder display-name -> (ascii glyph, hex color, category). The display names
# are exactly what toost's level decoder emits (see the inline notes for the
# handful that differ from the in-game name).
OBJ_META = {
    # terrain
    "Ground":              ("#", "#8B6914", CAT_TERRAIN),
    "Block":               ("B", "#C8A050", CAT_TERRAIN),
    "Hard Block":          ("H", "#888888", CAT_TERRAIN),
    "? Block":             ("?", "#F0C030", CAT_TERRAIN),
    "Hidden Block":        ("h", "#CCCCCC", CAT_TERRAIN),
    "Note Block":          ("N", "#E8A020", CAT_TERRAIN),
    "Donut Block":         ("d", "#F09050", CAT_TERRAIN),
    "Ice Block":           ("I", "#A0D8EF", CAT_TERRAIN),
    "P Block":             ("p", "#CC44CC", CAT_TERRAIN),
    "ON/OFF Block":        ("O", "#FF6600", CAT_TERRAIN),
    "Dotted-Line Block":   (".", "#AAAAAA", CAT_TERRAIN),
    "Blinking Block":      ("¤", "#FFAA00", CAT_TERRAIN),
    "Spike Block":         ("Ç", "#AA0000", CAT_TERRAIN),
    "Crate":               ("£", "#B87333", CAT_TERRAIN),
    "Stone":               ("H", "#888888", CAT_TERRAIN),  # decoder's actual name for Hard Block (id 6); "Stone Block" (id 75) never appears in practice
    "Goal Ground":         ("_", "#00AA00", CAT_TERRAIN),
    "Starting Brick":      ("{", "#C8A050", CAT_TERRAIN),
    "Castle Bridge":       ("·", "#885522", CAT_TERRAIN),
    "Tree":                ("¯", "#228B22", CAT_TERRAIN),
    "Slight Slope":        ("/", "#AA8833", CAT_TERRAIN),
    "Steep Slope":         ("\\","#CC9933", CAT_TERRAIN),
    # doors / warps
    "Pipe":                ("|", "#00BB00", CAT_DOOR),
    "Door":                ("D", "#4466FF", CAT_DOOR),
    "Warp Box":            ("§", "#6644FF", CAT_DOOR),
    "Key":                 ("±", "#FFD700", CAT_DOOR),
    "Checkpoint Flag":     ("f", "#00DDAA", CAT_DOOR),
    "Goal":                ("G", "#00FF44", CAT_DOOR),
    "Clear Pipe":          ("¢", "#44FFCC", CAT_DOOR),
    # enemies
    "Goomba":              ("g", "#CC6600", CAT_ENEMY),
    "Koopa":               ("K", "#44AA00", CAT_ENEMY),
    "Piranha Plant":       ("P", "#DD2200", CAT_ENEMY),
    "Piranha Flower":      ("P", "#DD2200", CAT_ENEMY),  # decoder's actual name for this object
    "Hammer Bro":          ("m", "#2244AA", CAT_ENEMY),
    "Thwomp":              ("t", "#6655AA", CAT_ENEMY),
    "Bob-omb":             ("o", "#444444", CAT_ENEMY),
    "Spiny":               ("s", "#CC2222", CAT_ENEMY),
    "Buzzy Beetle":        ("b", "#334488", CAT_ENEMY),
    "Lakitu":              ("L", "#DDAA00", CAT_ENEMY),
    "Lakitu's Cloud":      ("Â", "#CCCCAA", CAT_ENEMY),
    "Banzai Bill":         ("Z", "#333333", CAT_ENEMY),
    "Bullet Bill Blaster": ("V", "#333333", CAT_ENEMY),
    "Magikoopa":           ("y", "#8844CC", CAT_ENEMY),
    "Spike Top":           ("Å", "#AA3322", CAT_ENEMY),
    "Boo":                 ("u", "#DDDDDD", CAT_ENEMY),
    "Bowser":              ("X", "#BB3300", CAT_ENEMY),
    "Bowser Jr.":          ("x", "#CC5511", CAT_ENEMY),
    "Bowser Jr":           ("x", "#CC5511", CAT_ENEMY),  # decoder's actual name (no period)
    "Chain Chomp":         ("@", "#333333", CAT_ENEMY),
    "Cheep Cheep":         ("~", "#FF4488", CAT_ENEMY),
    "Blooper":             ("q", "#DDDDDD", CAT_ENEMY),
    "Wiggler":             ("w", "#AADD00", CAT_ENEMY),
    "Pokey":               ("Y", "#CCAA22", CAT_ENEMY),
    "Piranha Creeper":     ("¸", "#AA2200", CAT_ENEMY),
    "Porcupuffer":         ("µ", "#8866AA", CAT_ENEMY),
    "Fish Bone":           ("¿", "#AAAAAA", CAT_ENEMY),
    "Lava Bubble":         ("&", "#FF4400", CAT_ENEMY),
    "Rocky Wrench":        ("r", "#888844", CAT_ENEMY),
    "Muncher":             (",", "#00AA22", CAT_ENEMY),
    "Ant Trooper":         ("´", "#AA3300", CAT_ENEMY),
    "Monty Mole":          ("n", "#885522", CAT_ENEMY),
    "Mechakoopa":          ("R", "#666666", CAT_ENEMY),
    "Boom Boom":           ("!", "#BB4400", CAT_ENEMY),
    "Dry Bones":           ("9", "#BBBBAA", CAT_ENEMY),
    "Skipsqueak":          ("³", "#FFAA88", CAT_ENEMY),
    "Stingby":             ("É", "#DDCC00", CAT_ENEMY),
    "Angry Sun":           ("A", "#FF8800", CAT_ENEMY),
    "Charvaargh":          ("Ä", "#FF3300", CAT_ENEMY),
    "Bully":               ("Æ", "#883300", CAT_ENEMY),
    "Lemmy":               ("1", "#FF88CC", CAT_ENEMY),
    "Morton":              ("2", "#888888", CAT_ENEMY),
    "Larry":               ("3", "#44AA44", CAT_ENEMY),
    "Wendy":               ("4", "#FF44AA", CAT_ENEMY),
    "Iggy":                ("5", "#44AAFF", CAT_ENEMY),
    "Roy":                 ("6", "#AA44FF", CAT_ENEMY),
    "Ludwig":              ("7", "#4444CC", CAT_ENEMY),
    # items
    "Coin":                ("c", "#FFD700", CAT_ITEM),
    "Red Coin":            ("$", "#FF2200", CAT_ITEM),
    "1-Up Mushroom":       ("U", "#00CC00", CAT_ITEM),
    "1UP":                 ("U", "#00CC00", CAT_ITEM),  # decoder's actual name for this object
    "Fire Flower":         ("i", "#FF5500", CAT_ITEM),
    "Super Star":          ("*", "#FFFF00", CAT_ITEM),
    "Super Mushroom":      ("M", "#EE2222", CAT_ITEM),
    "Big Mushroom":        ("E", "#CC1111", CAT_ITEM),
    # Style Power-up A (id 44) gamestyle variants — see resolve_obj_name()
    "Super Leaf":          ("E", "#CC1111", CAT_ITEM),
    "Cape Feather":        ("E", "#CC1111", CAT_ITEM),
    "Propeller Mushroom":  ("E", "#CC1111", CAT_ITEM),
    "Super Hammer":        ("¬", "#996622", CAT_ITEM),
    "Big Coin":            ("£", "#FFAA00", CAT_ITEM),
    "P Switch":            ("S", "#4488FF", CAT_ITEM),
    "POW Block":           ("W", "#3366FF", CAT_ITEM),
    "POW":                 ("W", "#3366FF", CAT_ITEM),  # decoder's actual name (no "Block")
    "Spring":              ("J", "#DDDD00", CAT_ITEM),
    # Style Ride (id 45) gamestyle variants — see resolve_obj_name()
    "Goomba's Shoe":       ("z", "#CC6600", CAT_ITEM),
    "Yoshi's Egg":         ("z", "#CC6600", CAT_ITEM),
    "Cannon Box":          ("È", "#666666", CAT_ITEM),
    "Propeller Box":       ("}", "#8888FF", CAT_ITEM),
    "Goomba Mask":         ("Ê", "#CC6600", CAT_ITEM),
    "Bullet Bill Mask":    ("°", "#333333", CAT_ITEM),
    "Red POW Box":         ("²", "#FF3333", CAT_ITEM),
    # platforms
    "Lift":                ("-", "#DDAA55", CAT_PLATFORM),
    "Mushroom Platform":   ("T", "#FF6688", CAT_PLATFORM),
    "Semisolid Platform":  ("k", "#AAAAFF", CAT_PLATFORM),
    "Bridge":              ("=", "#AA8833", CAT_PLATFORM),
    "Lava Lift":           ("F", "#FF4400", CAT_PLATFORM),
    "Snake Block":         ("¹", "#44CC44", CAT_PLATFORM),
    "Track Block":         ("º", "#AA6622", CAT_PLATFORM),
    "Conveyor Belt":       ("»", "#888888", CAT_PLATFORM),
    "Fast Conveyor Belt":  ("¼", "#555555", CAT_PLATFORM),
    "Sprint Platform":     ("½", "#FF8800", CAT_PLATFORM),
    "Seesaw":              ("¾", "#AA8844", CAT_PLATFORM),
    "Swinging Claw":       ("j", "#AAAAAA", CAT_PLATFORM),
    "ON/OFF Trampoline":   ("À", "#FF6600", CAT_PLATFORM),
    "Mushroom Trampoline": ("Á", "#FF4488", CAT_PLATFORM),
    "Jumping Machine":     ("¦", "#8844FF", CAT_PLATFORM),
    "Half-Collision Platform": ("a", "#CCCCAA", CAT_PLATFORM),
    "Donut":               ("Ã", "#F09050", CAT_PLATFORM),
    # hazards
    "Fire Bar":            ("e", "#FF4400", CAT_HAZARD),
    "Saw":                 ("%", "#AAAAAA", CAT_HAZARD),
    "Burner":              ("l", "#FF6600", CAT_HAZARD),
    "Spikes":              ("^", "#888888", CAT_HAZARD),
    "Spike Ball":          ("v", "#884444", CAT_HAZARD),
    "Skewer":              ("0", "#666666", CAT_HAZARD),
    "Twister":             ("8", "#AADDFF", CAT_HAZARD),
    "Icicle":              ("Ë", "#AADDFF", CAT_HAZARD),
    # deco
    "Cloud":               ("<", "#CCCCFF", CAT_TERRAIN),
    "Vine":                ("[", "#00BB00", CAT_DECO),
    "Water Marker":        ("Î", "#0055FF", CAT_DECO),
    "Arrow":               ("Ï", "#FFFF00", CAT_DECO),
    "One-Way Wall":        ("]", "#FFFF88", CAT_DECO),
    "One-Way":             ("]", "#FFFF88", CAT_DECO),  # decoder's actual name (no "Wall")
    "Reel Camera":         ("Ñ", "#AAAAAA", CAT_DECO),
    "Sound Effect":        ("Ò", "#FFAAFF", CAT_DECO),
    # other
    "Player":              ("Ó", "#0000FF", CAT_OTHER),
    "Clown Car":           (";", "#FF4488", CAT_OTHER),
    "Koopa Clown Car":     ("Õ", "#44AA00", CAT_OTHER),
    "Track":               ("Ö", "#AAAAAA", CAT_OTHER),
    "Starting Arrow":      ("×", "#FFFF00", CAT_OTHER),
    "Cannon":              (")", "#444444", CAT_OTHER),
    "! Block":             ("Ù", "#FFAA00", CAT_OTHER),
    "_unknown":            ("?", "#FF00FF", CAT_OTHER),
}

GROUND_COLOR = "#8B6914"
GROUND_CHAR  = "#"

# ---------------------------------------------------------------------------
# Folds for objects whose OBJ_META glyph isn't a real mm2_tileset_we.json tile;
# without them the glyph collapses to the "unknown" tile at training time.
# ASCII_REPLACEMENTS approximates with a valid glyph, ASCII_DROP skips entirely.
# _check_tileset_coverage() enforces coverage at import.
# ---------------------------------------------------------------------------
ASCII_REPLACEMENTS = {
    "Spike Top":               "s",  # → Spiny
    "Fish Bone":               "~",  # → Cheep Cheep
    "Lakitu's Cloud":          ";",  # → Clown Car
    "Jumping Machine":         "J",  # → Spring
    "Mushroom Trampoline":     "J",  # → Spring
    "ON/OFF Trampoline":       "J",  # → Spring
    "Seesaw":                  "-",  # → Lift (no seesaw equivalent in SMMWE)
    "Muncher":                 "^",  # → Spikes (no wearing muncher on head in SMMWE)
    "P Block":                 "O",  # → ON/OFF block (closest togglable solid)
    "Spike Block":             "^",  # → Spikes
    "Goal Ground":             "#",  # → Ground
    "Starting Brick":          "#",  # → Ground (editor start marker)
    "Pokey":                   "s",  # → Spiny (nearest ground enemy)
    "Mechakoopa":              "K",  # → Koopa
    "Lemmy":                   "x",  # → Bowser Jr.
    "Morton":                  "x",  # → Bowser Jr.
    "Larry":                   "x",  # → Bowser Jr.
    "Wendy":                   "x",  # → Bowser Jr.
    "Iggy":                    "x",  # → Bowser Jr.
    "Roy":                     "x",  # → Bowser Jr.
    "Ludwig":                  "x",  # → Bowser Jr.
    "Red Coin":                "c",  # → Coin
    "Half-Collision Platform": "k",  # → Semisolid platform
    "Spike Ball":              "^",  # → Spikes
    "Icicle":                  "^",  # → Spikes (falling hazard)

    # Objects whose OBJ_META glyph is a leftover from the older, larger glyph
    # set; most are also filtered out during extraction.
    # terrain / blocks
    "Donut":                   "d",  # → Donut Block (same falling platform)
    "Blinking Block":          "O",  # → ON/OFF block (togglable solid)
    "Crate":                   "B",  # → Brick (breakable box)
    "Tree":                    "k",  # → Semisolid platform (stand on the canopy)
    "! Block":                 "B",  # → Brick (solid block flipped by the ! switch)
    # doors / warps
    "Warp Box":                "D",  # → Door (instant warp)
    "Clear Pipe":              "|",  # → Pipe (SM3DW pipe variant)
    # enemies
    "Piranha Creeper":         "P",  # → Piranha Plant
    "Porcupuffer":             "~",  # → Cheep Cheep (aquatic)
    "Ant Trooper":             "b",  # → Buzzy Beetle (armored ground walker)
    "Skipsqueak":              "g",  # → Goomba (small stompable walker)
    "Stingby":                 "L",  # → Lakitu (nearest flying enemy)
    "Charvaargh":              "&",  # → Lava Bubble (lava fire hazard)
    "Bully":                   "K",  # → Koopa (ground walker)
    # items / power-ups
    "Super Hammer":            "M",  # → Mushroom (power-up pickup)
    "Cannon Box":              "M",  # → Mushroom (wearable power-up box)
    "Propeller Box":           "M",  # → Mushroom (wearable power-up box)
    "Goomba Mask":             "M",  # → Mushroom (wearable power-up box)
    "Bullet Bill Mask":        "M",  # → Mushroom (wearable power-up box)
    "Red POW Box":             "W",  # → POW (a POW in a box)
    "Big Coin":                "c",  # → Coin (free path draws coins; covers the in-block path)
    # platforms — track/conveyor/snake levels are normally dropped upstream by
    "Sprint Platform":         "k",  # → Semisolid platform (SM3DW dash platform)
    "Koopa Clown Car":         ";",  # → Clown Car
    "Conveyor Belt":           "k",  # → Semisolid platform
    "Fast Conveyor Belt":      "k",  # → Semisolid platform
    "Snake Block":             "k",  # → Semisolid platform (moving path)
    "Track Block":             "k",  # → Semisolid platform (track-riding block)
    # Slopes are drawn as a ground staircase (see ascii.py); '#' records that.
    "Slight Slope":            "#",  # → Ground (staircase fill)
    "Steep Slope":             "#",  # → Ground (staircase fill)
}

# Editor/decorative markers with no tileset glyph: dropped from the grid.
ASCII_DROP = {
    "Castle Bridge",   # the goal-castle bridge is generated automatically
    "Key", "Arrow", "Water Marker", "Reel Camera", "Sound Effect",
    "Player", "Track", "Starting Arrow", 
}


def _check_tileset_coverage():
    """Fail at import if any drawable object lands on a glyph the tileset doesn't
    have -- it would otherwise become the unknown tile in training. Add the
    object to ASCII_REPLACEMENTS or ASCII_DROP to fix."""
    leaks = {}
    for name, (glyph, _color, _cat) in OBJ_META.items():
        if name == "_unknown" or name in ASCII_DROP:
            continue
        final = ASCII_REPLACEMENTS.get(name, glyph)
        if final not in TILESET:
            leaks[name] = final
    if leaks:
        listing = ", ".join(f"{n!r}->{g!r}" for n, g in sorted(leaks.items()))
        raise ValueError(
            f"OBJ_META objects with no tileset glyph: {listing}. Add each to "
            "ASCII_REPLACEMENTS (fold onto a tileset glyph) or ASCII_DROP (skip it)."
        )


_check_tileset_coverage()

CAT_COLORS = {
    CAT_TERRAIN:  "#C8A050",
    CAT_ENEMY:    "#CC4444",
    CAT_ITEM:     "#FFD700",
    CAT_PLATFORM: "#5599FF",
    CAT_DOOR:     "#44AAFF",
    CAT_HAZARD:   "#FF6600",
    CAT_DECO:     "#88BB88",
    CAT_OTHER:    "#AAAAAA",
}

# Style Power-up (id 44): the decoder emits a fixed name, but the actual
# power-up depends on gamestyle_raw.
STYLE_POWERUP_NAMES = {
    "Big Mushroom": {
        12621: "Big Mushroom",        # SMB1
        13133: "Super Leaf",          # SMB3
        22349: "Cape Feather",        # SMW
        21847: "Propeller Mushroom",  # NSMBU
    },
}

# Style Ride (id 45): a shoe in SMB1/SMB3, a Yoshi's Egg in SMW/NSMBU.
STYLE_RIDE_NAMES = {
    "Shoe Goomba": {
        12621: "Goomba's Shoe",
        13133: "Goomba's Shoe",
        22349: "Yoshi's Egg",
        21847: "Yoshi's Egg",
    },
}


def resolve_obj_name(obj_name: str, gamestyle_raw: int) -> str:
    """Gamestyle-correct name for the Style Power-up (44) / Ride (45) slots;
    pass-through otherwise."""
    for table in (STYLE_POWERUP_NAMES, STYLE_RIDE_NAMES):
        variants = table.get(obj_name)
        if variants:
            return variants.get(gamestyle_raw, obj_name)
    return obj_name


def get_meta(name: str):
    return OBJ_META.get(name, OBJ_META["_unknown"])


# ---------------------------------------------------------------------------
# Reverse direction: glyph -> decoder name, name -> MM2 object id
# ---------------------------------------------------------------------------

def build_char_to_name():
    """Invert OBJ_META to glyph -> name; first name per glyph wins. '_unknown'
    is skipped."""
    char_to_name = {}
    for name, (char, _color, _cat) in OBJ_META.items():
        if name == "_unknown":
            continue
        char_to_name.setdefault(char, name)
    return char_to_name


CHAR_TO_NAME = build_char_to_name()

# Object name -> MM2 object id. Variant aliases share a slot id. "Ground" is
# absent: '#' goes to the `ground` array, never `objects`.
NAME_TO_ID = {
    # terrain / blocks
    "Block": 4, "Hard Block": 6, "Stone": 6, "? Block": 5, "Hidden Block": 29,
    "Note Block": 23, "Donut Block": 21, "Ice Block": 63, "P Block": 79,
    "ON/OFF Block": 99, "Dotted-Line Block": 100, "Blinking Block": 108,
    "Spike Block": 110, "Crate": 112, "Goal Ground": 26, "Starting Brick": 37,
    "Castle Bridge": 49, "Tree": 106, "Slight Slope": 87, "Steep Slope": 88,
    # doors / warps
    "Pipe": 9, "Door": 55, "Warp Box": 97, "Key": 95, "Checkpoint Flag": 90,
    "Goal": 27, "Clear Pipe": 93,
    # enemies
    "Goomba": 0, "Koopa": 1, "Piranha Plant": 2, "Piranha Flower": 2,
    "Hammer Bro": 3, "Thwomp": 12, "Bob-omb": 15, "Spiny": 25,
    "Buzzy Beetle": 28, "Lakitu": 30, "Lakitu's Cloud": 31, "Banzai Bill": 32,
    "Bullet Bill Blaster": 13, "Magikoopa": 39, "Spike Top": 40, "Boo": 41,
    "Bowser": 62, "Bowser Jr.": 98, "Bowser Jr": 98, "Chain Chomp": 61,
    "Cheep Cheep": 56, "Blooper": 48, "Wiggler": 52, "Pokey": 78,
    "Piranha Creeper": 107, "Porcupuffer": 114, "Fish Bone": 103,
    "Lava Bubble": 60, "Rocky Wrench": 58, "Muncher": 57, "Ant Trooper": 96,
    "Monty Mole": 102, "Mechakoopa": 111, "Boom Boom": 77, "Dry Bones": 46,
    "Skipsqueak": 51, "Stingby": 65, "Angry Sun": 104, "Charvaargh": 86,
    "Bully": 117, "Lemmy": 120, "Morton": 121, "Larry": 122, "Wendy": 123,
    "Iggy": 124, "Roy": 125, "Ludwig": 126,
    # items
    "Coin": 8, "Red Coin": 92, "1-Up Mushroom": 33, "1UP": 33,
    "Fire Flower": 34, "Super Star": 35, "Super Mushroom": 20,
    "Big Mushroom": 44, "Super Leaf": 44, "Cape Feather": 44,
    "Propeller Mushroom": 44, "Super Hammer": 116, "Big Coin": 70,
    "P Switch": 18, "POW Block": 19, "POW": 19, "Spring": 10,
    "Goomba's Shoe": 45, "Yoshi's Egg": 45, "Cannon Box": 127,
    "Propeller Box": 128, "Goomba Mask": 129, "Bullet Bill Mask": 130,
    "Red POW Box": 131,
    # platforms
    "Lift": 11, "Mushroom Platform": 14, "Semisolid Platform": 16,
    "Bridge": 17, "Lava Lift": 36, "Snake Block": 84, "Track Block": 85,
    "Conveyor Belt": 94, "Fast Conveyor Belt": 53, "Sprint Platform": 80,
    "Seesaw": 91, "Swinging Claw": 105, "ON/OFF Trampoline": 132,
    "Mushroom Trampoline": 113, "Jumping Machine": 50,
    "Half-Collision Platform": 71, "Donut": 82,
    # hazards
    "Fire Bar": 24, "Saw": 68, "Burner": 54, "Spikes": 43, "Spike Ball": 74,
    "Skewer": 83, "Twister": 76, "Icicle": 118,
    # deco / other
    "Cloud": 22, "Vine": 64, "Water Marker": 101, "Arrow": 66,
    "One-Way Wall": 67, "One-Way": 67, "Reel Camera": 89, "Sound Effect": 109,
    "Player": 69, "Clown Car": 42, "Koopa Clown Car": 72, "Track": 59,
    "Starting Arrow": 38, "Cannon": 47, "! Block": 119,
}

# ---------------------------------------------------------------------------
# Items stored INSIDE a block
# ---------------------------------------------------------------------------
# A block stores its contained item's MM2 id in `cid`, not as a separate object.
# CONTAINER_BLOCK_IDS gates which blocks can carry one; CID_ITEM_NAME maps that
# id back to a name so the item can be drawn above the block (forward) or folded
# into `cid` (reverse). CID_ITEM_NAME takes the first item name per id.
CONTAINER_BLOCK_IDS = {4, 5, 29}  # Block (brick), ? Block, Hidden Block
CONTAINER_BLOCK_NAMES = {"Block", "? Block", "Hidden Block"}

CID_ITEM_NAME = {}
for _name, _oid in NAME_TO_ID.items():
    _meta = OBJ_META.get(_name)
    if _meta and _meta[2] == CAT_ITEM:
        CID_ITEM_NAME.setdefault(_oid, _name)


def contained_item_glyph(cid, gamestyle_raw):
    """Glyph for a block's contained item (cid), or None. Resolved like a
    free-standing object."""
    name = CID_ITEM_NAME.get(cid)
    if name is None:
        return None
    glyph = get_meta(resolve_obj_name(name, gamestyle_raw))[0]
    if name in ASCII_REPLACEMENTS:
        glyph = ASCII_REPLACEMENTS[name]
    return glyph


# ---------------------------------------------------------------------------
# Game style / theme maps (single copy for the whole pipeline;
# mm2pipeline_data.swe uses its own SWE-int variants).
# ---------------------------------------------------------------------------

GAMESTYLE_RAW = {
    "smb1": 12621, "smb3": 13133, "smw": 22349, "nsmbu": 21847, "sm3dw": 22323,
}
GAMESTYLE_NAME = {
    12621: "SMB1", 13133: "SMB3", 22349: "SMW", 21847: "NSMBU", 22323: "SM3DW",
}
THEME_RAW = {
    "overworld": 0, "underground": 1, "castle": 2, "airship": 3,
    "underwater": 4, "ghost": 5, "snow": 6, "desert": 7, "sky": 8, "forest": 9,
}
THEME_NAME = {
    0: "Ground", 1: "Underground", 2: "Castle", 3: "Airship", 4: "Underwater",
    5: "Ghost House", 6: "Snow", 7: "Desert", 8: "Sky", 9: "Forest",
}
