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
    58: "m",   # Tackle Fire (jumping)
    59: "n",   # Flying Shell/Mambu (flying)
    60: "o",   # Flying Shell/Mambu Spawner
    45: "p",   # Footholder (flying platform)
    63: "b",   # Bunby Heli (flying)
    18: "q",   # Kamadoma stand-in: the real Kamadoma isn't in Mega Man Maker, so the
               # closest jumping enemy (id 18) is mapped to the 'q' tile in its place.
}

# d == 6 (level objects / gimmick blocks): the 'e' subtype id -> VGLC char.
# The full Level-Objects id table is not publicly documented; only the ids
# confirmed from the .mmlv field reference and vglc_to_mmlv.py are decoded.
# Unknown level objects fall back to a generic solid block — this category is
# dominated by block-like obstacles, so '#' is the safest default (the most
# common unmapped id here is e=16, identity unknown).
GIMMICK_E_TO_CHAR = {
    9:  "B",   # 1x1 breakable block
    45: "B",   # 2x2 breakable block
    31: "M",   # moving platform
    5:  "A",   # appearing/disappearing block (verified against a labelled test level)
    54: "t",   # fake / secret transparent block (verified against a labelled test level)
    4:  "C",   # electric/hazard emitter ("extends a temporary passable damaging hazard outward")
    124:"I",   # Changkey fire-wave emitter (the vertical fire pillar 'I' tile)
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
    16: "M",   # Party Balloon (rideable transport)
}


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
        if dc == 4:                                 # player spawn (any character)
            return "P"
        if dc == 5:                                 # enemy
            if ei == 1:                             # Octopus Battery
                # 'g' holds a facing angle: 90/270 = vertical mover, else horizontal.
                g = cell.get("g")
                return "^" if (g is not None and int(g) in (90, 270)) else "<"
            return ENEMY_E_TO_CHAR.get(ei, "a")     # unknown enemy -> generic
        if dc == 6:                                 # level object / gimmick block
            return GIMMICK_E_TO_CHAR.get(ei, "#")
        if dc == 7:                                 # pickup (energy/life/1-up/tank …)
            return PICKUP_E_TO_CHAR.get(ei, "w")
        if dc == 8:                                 # level orb (e=15), boss door, or boss
            return BOSS_E_TO_CHAR.get(ei, "a")      # unmapped boss spawn -> generic enemy
        return "a"                                  # unknown object class

    # Tile layer ('i' field)
    if i is not None:
        return TILE_I_TO_CHAR.get(int(i))           # None for an unknown tile id

    # Water: real Mega Man Maker levels tag water cells with e=178 (no i/d), verified
    # against a labelled test level; 177 is also accepted for backward compatibility
    # with the value vglc_to_mmlv historically emitted.
    if cell.get("e") in (177.0, 178.0):
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
    """Convert one .mmlv to a 2-D list of VGLC chars."""
    cells = parse_mmlv(path)

    # Collapse the (layer, x, y) objects down to one char per (x, y). Process layers in
    # priority order and keep the first char placed at each coordinate, so a foreground
    # object never gets overwritten by a background one (e.g. water under an enemy).
    char_cells: Dict[Tuple[int,int], str] = {}
    for (_layer, tx, ty), cell in sorted(cells.items(), key=lambda kv: _layer_rank(kv[0][0])):
        ch = classify(cell)
        if ch is None or (tx, ty) in char_cells:
            continue
        char_cells[(tx, ty)] = ch

    if not char_cells:
        return []

    xs = [c[0] for c in char_cells]
    ys = [c[1] for c in char_cells]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

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

    return grid

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
    ap.add_argument("input",  help=".mmlv file or directory of .mmlv files")
    ap.add_argument("output", nargs="?", default=None,
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