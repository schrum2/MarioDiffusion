"""
mmlv_to_vglc.py  -  Convert Mega Man Maker .mmlv files to VGLC ASCII .txt

Format discoveries from reverse-engineering real .mmlv files:
  - Each placed object is a group of single-letter fields that all share one
    pixel coordinate:  fieldXpx,Ypx="value"  (one field per line).
  - Coordinates are in pixels (16 px per tile); only occupied cells appear.
  - IMPORTANT: a LEADING DIGIT before the field letter selects a different
    layer/section and must NOT be parsed as a placed object:
        0… global header     1… level settings (1a=name …)
        2… screen / scroll markers (2a,2b,2c,2d …)   4… author metadata
    The screen layer in particular emits 2d0,Y="40/50" entries; reading the
    'd' from those as an entity produced a phantom enemy in the top of the left column of
    almost every converted level.  parse_mmlv now anchors to the start of a
    line so only true object-layer fields (no digit prefix) are read.
  - MM Maker builds levels from discrete 16x14 screen chunks, and its screen grid is
    anchored at the world origin (screen corners fall on tile multiples of 16 in x and
    14 in y).  mmlv_to_grid exploits that: it snaps every object to its chunk and pads
    the bounding box out to whole chunks, so the ASCII always tiles into 16x14 chunks
    that line up with what the player sees -- including protrusions, which claim their
    own padded chunk above/below the main band rather than shifting everything.  The
    '2' screen markers (2a enabled / 2d background) are deliberately NOT used to crop:
    2a-vs-content differed for <0.1% of screens across the corpus, and 2d is only the
    painted subset of the playable area, so cropping by it deletes large unpainted-but-
    built regions.  Content alone defines the crop.
  - Object-layer fields we use:
      i = tile id:    1=solid block, 2=spike, 3=ladder
      d = object class: 4=player spawn, 5=enemy, 6=gimmick/block, 7=pickup,
          8=boss / boss door / level-exit orb (orb is e=15)
      e = subtype id  (enemy id when d=5, gimmick id when d=6, item id when d=7)
      g = orientation flag (distinguishes the vertical vs horizontal Suzy)
      a = "exists" flag present on every object (ignored); j,k,l = textures (ignored)

The decode tables below are the inverse of megaman/vglc_to_mmlv.py; the
project's source of truth for the VGLC-char <-> Mega Man Maker encoding; so the
two scripts round-trip and the output covers the full MM.json tileset rather
than collapsing every enemy/gimmick/collectible to a single character.
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

TILE_PX = 16  # pixels per tile
MEGAMAN_SCENE_HEIGHT = 14   # actual playable vertical scene height
# Add near the top with other constants
MEGAMAN_PLAYABLE_HEIGHT = 14  # Actual playable scene height (distinct from
                                # the 16-tile nav window used elsewhere in the
                                # repo for screen-based navigation — do not
                                # change that 16, this is a separate constant)
MEGAMAN_SCREEN_WIDTH = 16  # Mega Man Maker's screen grid width, used to scan
                            # the level in screen-sized blocks when deciding
                            # which empty cells are walkable sky vs truly outside any screen


# Object field regex.  The leading '^' (with re.MULTILINE) keeps us on true
# object lines and skips the digit-prefixed section lines ('2d0,…', '1a…', '4a…')
# that otherwise leak a phantom enemy into the level.
#
# An optional 'z<NN>' prefix (e.g. 'z20o3376,3088=…', 'z60e3216,3088=…') marks an
# object placed on a non-default LAYER.  Newer Mega Man Maker versions (1.10+) put
# things like water (z20) and some enemies/gimmicks (z60) on these layers; the old
# regex matched only the default (prefix-less) layer and silently dropped every
# layered object -- which is why water never appeared in the converted data. We now
# capture the optional layer in group 1 and key cells by (layer, x, y) so objects on
# different layers at the same coordinate stay separate instead of merging fields.
_FIELD_RE = re.compile(r'^(z\d+)?([a-z])(\d+),(\d+)="([^"]+)"', re.MULTILINE)


def parse_mmlv(path: Path) -> Dict[Tuple[str,int,int], dict]:
    """Return sparse dict of (layer, tile_x, tile_y) -> {field: float_value}.

    'layer' is "" for the default/main layer or the raw prefix (e.g. "z20") for a
    layered object. mmlv_to_grid collapses the layers back down to one char per cell.
    """
    text = path.read_bytes().decode("utf-8", errors="replace").replace('\r', '')
    cells: Dict[Tuple[str,int,int], dict] = {}
    for m in _FIELD_RE.finditer(text):
        layer = m.group(1) or ""
        field = m.group(2)
        tx = int(m.group(3)) // TILE_PX
        ty = int(m.group(4)) // TILE_PX
        val = float(m.group(5))
        cells.setdefault((layer, tx, ty), {})[field] = val
    return cells


#  decode tables (inverse of megaman/vglc_to_mmlv.py)

# Tile layer: the 'i' field id -> VGLC char.
TILE_I_TO_CHAR = {1: "#", 2: "H", 3: "|"}

# d == 5 (enemies): the 'e' subtype id -> VGLC char.  Enemy ids are NOT
# sequential by series order (e.g. Spine is 48, not 6), so only ids verified
# against the .mmlv field reference / vglc_to_mmlv.py are listed here; every
# other enemy id falls back to the generic enemy 'a'.
ENEMY_E_TO_CHAR = {
    0:  "a",   # Met (ground, ranged)
    1:  "<",   # Octopus Battery - horizontal ('^' when g marks up/down; see classify)
    2:  "c",   # Beak (ranged, wall-mounted)
    3:  "d",   # Picketman (ranged, shielded)
    4:  "e",   # Screw Bomber (stationary, ranged)
    5:  "f",   # Big Eye (jumping)
    7:  "r",   # Sniper Joe (shielded, ranged)
    48: "g",   # Spine / Gabyoall (ground)
    49: "h",   # Crazy Razy (ground)
    52: "i",   # Watcher (floating, ranged)
    56: "j",   # Killer Bullet (flying)
    57: "k",   # Killer Bullet Spawner
    58: "I",   # Tackle Fire enemy -> the 'I' fire tile. ('m' is Bombombomb, which has
               # no Mega Man Maker equivalent, so nothing here maps to it.)
    59: "n",   # Flying Shell/Mambu (flying)
    60: "o",   # Flying Shell/Mambu Spawner
    45: "p",   # Footholder (flying platform)
    # Bunby Heli (e63) and Kamadoma (e18) are intentionally NOT mapped: neither is part
    # of the MMLV tileset, so if one is ever encountered it falls back to the generic
    # enemy 'a' rather than emitting a char the MMLV tileset doesn't define.
}

# d == 6 (level objects / gimmick blocks): the 'e' subtype id -> VGLC char.
# The full Level-Objects id table is not publicly documented; only the ids
# confirmed from the .mmlv field reference and vglc_to_mmlv.py are decoded.
# Unknown level objects fall back to a generic solid block — this category is
# dominated by block-like obstacles, so '#' is the safest default (the most
# common unmapped id here is e=16, identity unknown).
GIMMICK_E_TO_CHAR = {
    9:  "B",   # 1x1 breakable block
    45: "B",   # 2x2 breakable block (see TWO_BY_TWO_E_IDS: expands to a full 2x2)
    93: "B",   # 2x2 breakable block, another variant (see TWO_BY_TWO_E_IDS)
    205:"B",   # 2x2 breakable block, another variant (see TWO_BY_TWO_E_IDS)
    206:"B",   # 2x2 breakable block, another variant (see TWO_BY_TWO_E_IDS)
    186:"#",   # 2x2 UNbreakable solid block (see TWO_BY_TWO_E_IDS: expands to a full 2x2 of '#')
    208:"B",   # 1-wide x 2-tall vertical breakable wall (see TWO_TALL_E_IDS: expands one tile up)
    27: "B",   # 2x2 weapon-specific breakable block. Every weapon variant shares this one
               # e id; the required weapon is in the 'o' field (o=1..8 special weapons,
               # o=9999 default, absent = unassigned), so a single mapping covers them all.
    31: "M",   # moving platform (see MOVING_PLATFORM_E_IDS: only the path node hosting the
               # physical platform decodes to 'M'; the bare path/track nodes decode to air)
    262:"M",   # moving platform, another variant (see MOVING_PLATFORM_E_IDS)
    5:  "A",   # appearing/disappearing block (verified against a labelled test level)
    54: "t",   # fake / secret transparent block (verified against a labelled test level)
    163:"C",   # electric/hazard emitter ("extends a temporary passable damaging hazard outward").
               # Verified d6/e163 against a game-authored test level (three emitters flanking two
               # water pools + one atop the middle pillar). The earlier e4 id was a misidentification.
    73: ">",   # conveyor belt (see CONVEYOR_E_IDS; direction resolved in classify(): 'b'=-1 -> 'E', else '>')
    74: ">",   # conveyor belt, another variant (see CONVEYOR_E_IDS)
    124:"I",   # Changkey fire spawner (reuses the tackle-fire sprite; the 'I' fire tile)
    11: "F",   # falling platform: a solid block that drops when stood on. Verified d6/e11
    43: "x",   # fan: blows Mega Man upward.
    13: "s",   # spring: bounces Mega Man upward when touched. Verified d6/e13 against a labelled
               # test level. (Distinct from d5 e13, an unrelated unidentified enemy id.)
    266:"T",   # teleporter (paired warp gimmick).
    65: "T",   # teleporter, another variant (same m/n partner-link + f style fields as e266).
    267:"#",   # 2-wide horizontal solid block (see TWO_WIDE_E_IDS: expands one tile left)
    261:"#",   # 2-wide horizontal solid platform, another variant (see TWO_WIDE_E_IDS)
}

# d == 7 (pickups): the 'e' subtype id -> VGLC char.  Pickup ids are a small
# sequential set (verified contiguous 0..7 in real levels) that matches the
# ordered list on the MegaManMaker wiki Pickups page, so the energy/life/1-up
# items map onto the corresponding MM.json powerup characters.  Items with no
# dedicated tileset character collapse
# to the closest powerup, defaulting to small weapon energy 'w'.
PICKUP_E_TO_CHAR = {
    0: "L",   # Large Health  (Large Life Energy)
    1: "l",   # Small Health  (Small Life Energy)
    2: "W",   # Large Weapon Energy
    3: "w",   # Small Weapon Energy
    4: "+",   # Life / 1-Up extra life
    5: "L",   # E-Tank   (full health -> generic large life)
    6: "*",   # M-Tank   (full restore -> Yashichi-equivalent)
    12: "*",  # Yashichi (full restore)
}

# d == 8 (bosses / boss doors / level orb): the 'e' subtype id -> VGLC char,
# from the boss table in the .mmlv format reference.  Only id 15 is the actual
# level-exit orb, so 'Z' (the MM.json "level exit / final goal") is reserved for
# it; boss doors decode to the passable-door tile and the boss spawns themselves
# decode to a generic enemy.
BOSS_E_TO_CHAR = {
    15: "Z",   # Energy Element / MM1 Exit Orb  -> the level exit
    0:  "D",   # Vertical Boss Door (also the no-'e' default)
    1:  "D",   # Horizontal Boss Door
    33: "D",   # Vertical Boss Door: a 2-wide x 4-tall door block (see BOSS_DOOR_V_E_IDS:
               # expands its bottom-right anchor left + 3 up into the full 2x4 footprint)
    34: "D",   # Horizontal Boss Door: a 4-wide x 2-tall door block (see BOSS_DOOR_H_E_IDS:
               # anchor is the bottom row, one tile left of the right edge -> expands 2 left,
               # 1 right, and 1 up into the full 4x2 footprint)
    16: "Z",   # Confetti Balloon: ends the level when shot, so it maps to the level-exit/orb
               # tile 'Z' (same as the e15 exit orb). Verified d8/e16 against a labelled level;
               # the earlier "Party Balloon / rideable transport -> M" id was a misidentification.
}

# Water / liquid tiles: a water cell carries only an 'e' id (no d/i), and Mega Man Maker
# uses a distinct id for every liquid family (water/acid/lava/oil/...) and each surface vs
# extends-downward variant. All of them collapse to the single water tile '~'. These ids
# were captured from a labelled test level containing one of every water tile type and
# nothing else; the contiguous clusters are the different families/variants.
WATER_E_IDS = (
    set(range(177, 195))    # 177-194
    | set(range(621, 629))  # 621-628
    | set(range(1153, 1164))  # 1153-1163
    | {1210, 1211, 1687}
)

# Lava: like water, lava is a liquid cell carrying only an 'e' id, but it is a damaging
# hazard, so it maps to its own tile '!' (a solid/hazard, like spikes) rather than the
# passable water '~'. Lava has one id per surface/body variant, captured contiguous
# (1095-1102) from a labelled test level containing one of every lava tile type.
LAVA_E_IDS = set(range(1095, 1103))  # 1095-1102


# d == 6 gimmick ids that are 2x2 blocks. Mega Man Maker stores a 2x2 block as a single
# object whose coordinate is the block's BOTTOM-RIGHT tile, so on its own it decodes to
# just that one cell and the other three tiles read as gaps. mmlv_to_grid expands each of
# these to the full 2x2 by also filling the tiles directly above, directly left, and
# diagonally up-left with the same char.
TWO_BY_TWO_E_IDS = {27, 45, 93, 205, 206, 186}

# d == 6 gimmick ids that are 2-wide x 1-tall horizontal blocks. Like the 2x2 blocks these
# are stored as a single object at the block's RIGHT tile, so on their own they decode to
# just that one cell and the left tile reads as a gap. mmlv_to_grid expands each to the
# full 2x1 by also filling the tile directly to the left with the same char.
TWO_WIDE_E_IDS = {261, 267}

# d == 6 gimmick ids that are 1-wide x 2-tall vertical blocks. Like the other multi-tile
# blocks these are stored as a single object, here at the block's BOTTOM tile, so on their
# own they decode to just that one cell and the tile above reads as a gap. mmlv_to_grid
# expands each to the full 1x2 by also filling the tile directly ABOVE with the same char.
TWO_TALL_E_IDS = {208}

# d == 6 gimmick ids that are conveyor belts. Every conveyor of a given type shares one e
# id (only the belt-art fields f/p differ); classify() reads the 'b' field for the push
# direction ('b'=-1 -> left 'E', else right '>'). Multiple conveyor types exist (e73, e74).
CONVEYOR_E_IDS = {73, 74}

# Invisible logic / trigger objects that have NO tile representation and should be ignored
# (decoded to empty air '-'), e.g. boss event triggers. These are keyed by the full (d, e) pair,
# NOT by 'e' alone, because subtype ids collide across classes -- e.g. d8/e36 is the boss event
# trigger, but d6/e36 is a common (still unidentified) block-like gimmick, so an 'e'-only ignore
# set would wrongly blank out every d6/e36. classify() returns None (air) for any (d, e) here.
TRIGGER_IDS = {(8, 36)}   # d8 e36 = boss event trigger

# d == 8 (boss category) boss-door blocks. Like the multi-tile gimmick blocks these are stored
# as a single object at one corner, so on their own they decode to just that one cell and the
# rest of the footprint reads as gaps. mmlv_to_grid expands them to the full door. Note these are
# the d8 boss class, NOT the d6 block class the other footprint sets use. Two orientations:
#   VERTICAL   (e33): 2-wide x 4-tall, anchored at the BOTTOM-RIGHT tile.
#   HORIZONTAL (e34): 4-wide x 2-tall, anchored at the BOTTOM row, one tile LEFT of the right
#                     edge (relative tile (row=1, col=2) of the 0-indexed 4x2 block).
BOSS_DOOR_V_E_IDS = {33}
BOSS_DOOR_H_E_IDS = {34}

# d == 6 gimmick ids that are moving platforms. A moving platform is placed as a chain of
# invisible PATH/track nodes that all share the same e id; exactly one node (the platform's
# origin) additionally carries the 'h' field and is where the physical, ridable platform
# actually sits. classify() therefore decodes only the 'h'-bearing node to the solid moving
# tile 'M' and the bare path/track nodes to the passable path tile '=', so a long path reads as
# a run of '=' with a single 'M' rather than a solid row of platforms. (Verified for e31 from
# Claude.mmlv: a 9-tile path with the platform on the single node carrying h=2; e262 is the
# other platform variant, assumed to use the same field.)
MOVING_PLATFORM_E_IDS = {31, 262}


def is_2x2_block(cell: dict) -> bool:
    """True if the cell is a d6 gimmick that occupies a 2x2 tile footprint."""
    d = cell.get("d")
    e = cell.get("e")
    return d is not None and int(d) == 6 and e is not None and int(e) in TWO_BY_TWO_E_IDS


def is_2x1_block(cell: dict) -> bool:
    """True if the cell is a d6 gimmick that occupies a 2-wide x 1-tall footprint."""
    d = cell.get("d")
    e = cell.get("e")
    return d is not None and int(d) == 6 and e is not None and int(e) in TWO_WIDE_E_IDS


def is_1x2_block(cell: dict) -> bool:
    """True if the cell is a d6 gimmick that occupies a 1-wide x 2-tall footprint."""
    d = cell.get("d")
    e = cell.get("e")
    return d is not None and int(d) == 6 and e is not None and int(e) in TWO_TALL_E_IDS


def is_boss_door_v(cell: dict) -> bool:
    """True if the cell is a d8 vertical boss door (2-wide x 4-tall footprint)."""
    d = cell.get("d")
    e = cell.get("e")
    return d is not None and int(d) == 8 and e is not None and int(e) in BOSS_DOOR_V_E_IDS


def is_boss_door_h(cell: dict) -> bool:
    """True if the cell is a d8 horizontal boss door (4-wide x 2-tall footprint)."""
    d = cell.get("d")
    e = cell.get("e")
    return d is not None and int(d) == 8 and e is not None and int(e) in BOSS_DOOR_H_E_IDS


def classify(cell: dict) -> str:
    """Map a cell's object-layer fields to a VGLC character (or None for air)."""
    d = cell.get("d")
    i = cell.get("i")

    # Object class ('d') takes priority over the tile layer; in real levels a
    # cell never carries both an 'i' and a 'd'.  A missing 'e' field means the
    # default subtype 0 (per the .mmlv "absent value == default" rule).
    if d is not None:
        dc = int(d)
        e = cell.get("e")
        ei = int(e) if e is not None else 0
        if (dc, ei) in TRIGGER_IDS:                 # invisible logic/trigger object: no tile,
            return None                             # ignore it and leave the cell as empty air
        if dc == 4:                                 # player spawn (any character)
            return "P"
        if dc == 5:                                 # enemy
            if ei == 1:                             # Octopus Battery
                # 'g' holds a facing angle: 90/270 = vertical mover, else horizontal.
                g = cell.get("g")
                return "^" if (g is not None and int(g) in (90, 270)) else "<"
            return ENEMY_E_TO_CHAR.get(ei, "a")     # unknown enemy -> generic
        if dc == 6:                                 # level object / gimmick block
            if ei in MOVING_PLATFORM_E_IDS:         # moving platform path: only the node with
                # the physical platform carries the 'h' field; it decodes to the solid moving
                # tile 'M', while the bare path/track nodes (no 'h') decode to the path tile '='.
                return "M" if ("h" in cell) else "="
            if ei in CONVEYOR_E_IDS:                # conveyor belt: 'b'=-1 faces left, else right.
                # Within a conveyor family only the belt-art fields (f/p) differ, so gate on
                # the id and read 'b' for the push direction. Multiple conveyor types exist
                # (e73, e74, ...), all sharing this same left/right encoding.
                b = cell.get("b")
                return "E" if (b is not None and int(b) == -1) else ">"
            return GIMMICK_E_TO_CHAR.get(ei, "#")
        if dc == 7:                                 # pickup (energy/life/1-up/tank …)
            return PICKUP_E_TO_CHAR.get(ei, "w")
        if dc == 8:                                 # level orb (e=15), boss door, or boss
            return BOSS_E_TO_CHAR.get(ei, "a")      # unmapped boss spawn -> generic enemy
        return "a"                                  # unknown object class

    # Tile layer ('i' field)
    if i is not None:
        return TILE_I_TO_CHAR.get(int(i))           # None for an unknown tile id

    # Liquids: a liquid cell carries only an 'e' id (no i/d). Water collapses to the
    # passable '~'; lava is a damaging hazard, so it collapses to the spike tile 'H'.
    e = cell.get("e")
    if e is not None:
        ei = int(e)
        if ei in LAVA_E_IDS:
            return "!"
        if ei in WATER_E_IDS:
            return "~"

    # Cell exists but carries no recognised tile/object field.
    return None

def fill_walkable_by_screen_grid(grid, char_cells, min_x, min_y, width, height):
    """
    Scan the level in MEGAMAN_PLAYABLE_HEIGHT x MEGAMAN_SCREEN_WIDTH blocks.
    If a block contains any real placed tile, every '@' cell in that block
    becomes '-' (walkable sky). If a block has no tiles at all, it's left
    as '@' (truly outside any screen).
    """
    occupied = set()
    for (tx, ty) in char_cells.keys():
        occupied.add((ty - min_y, tx - min_x))

    for block_row_start in range(0, height, MEGAMAN_PLAYABLE_HEIGHT):
        for block_col_start in range(0, width, MEGAMAN_SCREEN_WIDTH):
            block_row_end = min(block_row_start + MEGAMAN_PLAYABLE_HEIGHT, height)
            block_col_end = min(block_col_start + MEGAMAN_SCREEN_WIDTH, width)

            has_content = any(
                (r, c) in occupied
                for r in range(block_row_start, block_row_end)
                for c in range(block_col_start, block_col_end)
            )

            if has_content:
                for r in range(block_row_start, block_row_end):
                    for c in range(block_col_start, block_col_end):
                        if grid[r][c] == '@':
                            grid[r][c] = '-'

    return grid

def _layer_rank(layer: str) -> int:
    """Placement priority for collapsing layered objects onto one grid cell.

    Lower rank wins (placed first, never overwritten). The default/main layer ("")
    holds the primary gameplay objects and wins outright; higher z-layers (z60) hold
    foreground enemies/gimmicks and beat lower z-layers (z20) such as the water
    background, so an enemy standing in water shows the enemy, not the water.
    """
    if layer == "":
        return 0
    return 1000 - int(layer[1:])


def mmlv_to_grid(path: Path):
    """Convert one .mmlv to a 2-D list of VGLC chars.

    The output is always a whole number of 16x14 screen chunks.  We take the bounding
    box of every chunk that holds an object and pad it out to whole chunks, so no tile
    is ever dropped -- content that protrudes one screen past the play band just claims
    its own padded chunk, keeping 16x14 chunk scans aligned to what the game shows.
    """
    path = Path(path)
    cells = parse_mmlv(path)

    # Collapse the (layer, x, y) objects down to one char per (x, y). Process layers in
    # priority order and keep the first char placed at each coordinate, so a foreground
    # object never gets overwritten by a background one (e.g. water under an enemy).
    char_cells: Dict[Tuple[int,int], str] = {}
    for (_layer, tx, ty), cell in sorted(cells.items(), key=lambda kv: _layer_rank(kv[0][0])):
        ch = classify(cell)
        if ch is None:
            continue
        if (tx, ty) not in char_cells:
            char_cells[(tx, ty)] = ch
        # A 2x2 block is stored as one object at its bottom-right tile; fill the other
        # three tiles (up, left, up-left) with the same char. setdefault keeps any
        # higher-priority object already placed there (e.g. a foreground enemy).
        if is_2x2_block(cell):
            for nx, ny in ((tx - 1, ty), (tx, ty - 1), (tx - 1, ty - 1)):
                char_cells.setdefault((nx, ny), ch)
        # A 2-wide horizontal block is stored at its right tile; fill the tile to its left.
        elif is_2x1_block(cell):
            char_cells.setdefault((tx - 1, ty), ch)
        # A 1-wide x 2-tall block is stored at its bottom tile; fill the tile directly above.
        elif is_1x2_block(cell):
            char_cells.setdefault((tx, ty - 1), ch)
        # A vertical boss door is a 2-wide x 4-tall block stored at its bottom-right tile; fill
        # the other seven tiles of the 2x4 footprint (one column left, three rows up).
        elif is_boss_door_v(cell):
            for dx in (0, -1):
                for dy in (0, -1, -2, -3):
                    if dx == 0 and dy == 0:
                        continue
                    char_cells.setdefault((tx + dx, ty + dy), ch)
        # A horizontal boss door is a 4-wide x 2-tall block stored at the bottom row, one tile
        # left of the right edge; fill the other seven tiles (two columns left, one right, one up).
        elif is_boss_door_h(cell):
            for dx in (-2, -1, 0, 1):
                for dy in (0, -1):
                    if dx == 0 and dy == 0:
                        continue
                    char_cells.setdefault((tx + dx, ty + dy), ch)

    if not char_cells:
        return []

    # Crop rectangle (inclusive tile coords), snapped to whole 16x14 screen chunks.
    # Snap each object to the top-left tile of its screen chunk and take the bounding
    # box over that set: this pads out to whole chunks so a 1-tile overhang past a
    # chunk border simply claims its own chunk, and the grid always tiles cleanly.
    # (MM Maker anchors its screen grid at the origin, so //16 and //14 land on real
    # screen boundaries -- which is what keeps 16x14 chunk scans aligned to the game.)
    crop_screens = {((tx // MEGAMAN_SCREEN_WIDTH) * MEGAMAN_SCREEN_WIDTH,
                     (ty // MEGAMAN_PLAYABLE_HEIGHT) * MEGAMAN_PLAYABLE_HEIGHT)
                    for (tx, ty) in char_cells}
    sx = [s[0] for s in crop_screens]
    sy = [s[1] for s in crop_screens]
    min_x, max_x = min(sx), max(sx) + (MEGAMAN_SCREEN_WIDTH - 1)
    min_y, max_y = min(sy), max(sy) + (MEGAMAN_PLAYABLE_HEIGHT - 1)

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # Start everything as inaccessible space
    grid = [['@'] * width for _ in range(height)]

    grid = fill_walkable_by_screen_grid(grid, char_cells, min_x, min_y, width, height)

    # Place actual tiles/entities (overwrites any '-' fill at that position)
    for (tx, ty), ch in char_cells.items():
        row = ty - min_y
        col = tx - min_x
        grid[row][col] = ch

    flood_liquids_down(grid)

    return grid


# Liquid tiles that fall to fill air beneath them (water '~' and lava '!').
LIQUID_CHARS = ("~", "!")


def flood_liquids_down(grid) -> None:
    """Flood liquids downward: any air cell directly beneath a liquid becomes that liquid.

    Mega Man Maker only stores the tiles the author painted, so a deep pool often keeps
    just its surface row(s) of liquid with empty (air) cells underneath. In the VGLC grid
    that reads as liquid floating over a hole. Here we let each column's liquid fall:
    scanning top-to-bottom, once a liquid char is seen every contiguous air cell below it
    ('-' walkable sky or '@' outside-screen) is filled with that same liquid, until a
    solid/other tile stops the flow. A different liquid switches which liquid is falling
    (e.g. water beneath lava keeps flooding, now as water), so water '~' and lava '!'
    each pool correctly.
    """
    if not grid:
        return
    AIR = ("-", "@")
    height = len(grid)
    width = len(grid[0])
    for col in range(width):
        liquid = None
        for row in range(height):
            ch = grid[row][col]
            if ch in LIQUID_CHARS:
                liquid = ch
            elif liquid is not None and ch in AIR:
                grid[row][col] = liquid
            else:
                liquid = None

def mmlv_to_vglc(path: Path) -> list[str]:
    """Convert one .mmlv to a list of VGLC ASCII row strings.

    Thin wrapper around mmlv_to_grid used by bulk_mmlv_to_vglc.py.
    """
    grid = mmlv_to_grid(Path(path))
    return ["".join(row) for row in grid]


def convert(src: Path, dst: Path):
    grid = mmlv_to_grid(src)
    if not grid:
        print(f"  [skip] {src.name} - no content found")
        return False
    dst.write_text("\n".join("".join(row) for row in grid) + "\n", encoding="utf-8")
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    print(f"  {src.name} -> {dst.name}  ({cols}w x {rows}h)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Convert .mmlv → VGLC .txt")
    ap.add_argument("--input",  help=".mmlv file or directory of .mmlv files")
    ap.add_argument("--output", nargs="?", default=None,
                    help="Output directory for .txt files (default: 'txt/' next to input)")
    args = ap.parse_args()

    src_path = Path(args.input)

    # Default output: a 'txt' folder next to the input file/dir
    if args.output is None:
        base = src_path.parent if src_path.is_file() else src_path
        out_dir = base / "txt"
    else:
        out_dir = Path(args.output)

    out_dir.mkdir(parents=True, exist_ok=True)

    if src_path.is_dir():
        files = sorted(src_path.glob("*.mmlv"))
    else:
        files = [src_path]

    ok = 0
    for f in files:
        dst = out_dir / f.with_suffix(".txt").name
        if convert(f, dst):
            ok += 1
    print(f"\nConverted {ok}/{len(files)} levels.")
    print(f"Output saved to: {out_dir}")


if __name__ == "__main__":
    main()