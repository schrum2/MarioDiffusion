"""Convert between MM2 level JSON (toost's export) and the ASCII grid the
diffusion model trains on.

Usage:
    python -m mm2pipeline_data.ascii to-ascii --input <json_folder> --output_folder <ascii_folder>
    python -m mm2pipeline_data.ascii to-json  --input <txt file or folder> --output_folder <json_folder>

Object metadata comes from mm2pipeline_data.tiles. Both directions are lossy: the
forward path folds off-tileset glyphs (ASCII_REPLACEMENTS / ASCII_DROP), the
reverse path rebuilds baseline objects via coalesce() plus semisolid and
goal repair.
"""
import json
import os
import sys
import argparse
from pathlib import Path

from .tiles import (
    GROUND_CHAR,
    ASCII_REPLACEMENTS,
    ASCII_DROP,
    CHAR_TO_NAME,
    NAME_TO_ID,
    GAMESTYLE_RAW,
    GAMESTYLE_NAME,
    THEME_RAW,
    THEME_NAME,
    CAT_ITEM,
    OBJ_META,
    CONTAINER_BLOCK_IDS,
    CONTAINER_BLOCK_NAMES,
    contained_item_glyph,
    resolve_obj_name,
    get_meta,
)

# ===========================================================================
# Forward: JSON -> ASCII
# ===========================================================================

# ---------------------------------------------------------------------------
# Pipe direction helpers (flag % 0x80: 0x00=R, 0x20=L, 0x40=U, 0x60=D)
# ---------------------------------------------------------------------------
def _pipe_direction(flag: int) -> str:
    d = flag % 0x80
    if d == 0x00: return 'R'
    if d == 0x20: return 'L'
    if d == 0x40: return 'U'
    return 'D'

_PIPE_DIR_CHAR = {'R': '→', 'L': '←', 'U': '↑', 'D': '↓'}


# ---------------------------------------------------------------------------
# Tile size helper — uses w/h from JSON directly (already tile counts)
# ---------------------------------------------------------------------------
# toost draws these at a fixed size no matter the w/h (from LevelDrawer.cpp).
_FIXED_SIZE = {
    "Clown Car": (2, 2),
}


def obj_tile_size(obj: dict):
    """(w, h) in tiles. Pipes use h as length regardless of direction; the
    cross-section is always 2."""
    fixed = _FIXED_SIZE.get(obj.get("name"))
    if fixed:
        return fixed
    if obj.get("name") == "Pipe":
        direction = _pipe_direction(obj.get("flag", 0))
        length = max(1, obj.get("h", 1))
        if direction in ('U', 'D'):
            return 2, length
        else:
            return length, 2
    w = max(1, obj.get("w", 1))
    h = max(1, obj.get("h", 1))
    return w, h


# Objects whose x is the left-tile center (col = x // 160); everything else
# stores the center of its span (col = x//160 - w//2).
_LEFT_ANCHOR = frozenset({
    "Pipe",
    "Bridge",
    "Conveyor Belt",
    "Fast Conveyor Belt",
    "Mushroom Platform",
    "Semisolid Platform",
    "Slight Slope",
    "Steep Slope",
    "Half-Collision Platform",
    # synthetic objects injected by _normalize_level use tile coords directly
    "Ground",
    "Starting Brick",
    "Goal",
})


def obj_anchor(obj: dict):
    """(col, row) of the object's bottom-left tile. Left-anchor objects use
    col = x // 160, everything else col = x//160 - w//2; row = y // 160.
    Pipes get direction-specific adjustments from the C++ renderer."""
    if obj.get("name") == "Pipe":
        direction = _pipe_direction(obj.get("flag", 0))
        base_col = obj["x"] // 160
        base_row = obj["y"] // 160
        w, h = obj_tile_size(obj)
        if direction == 'U':
            # columns [col, col+1], rows [base_row, base_row+h-1]
            return base_col, base_row
        elif direction == 'D':
            # x offset -1 tile; pipe extends downward (decreasing row)
            return base_col - 1, base_row - h + 1
        elif direction == 'R':
            # columns [col, col+w-1], rows [base_row, base_row]
            return base_col, base_row - 1
        else:  # L
            # pipe extends left; y stays the same
            return base_col - w + 1, base_row

    w, h = obj_tile_size(obj)
    x = obj["x"]
    if obj.get("name", "") in _LEFT_ANCHOR:
        col = x // 160
    else:
        col = x // 160 - w // 2
    row = obj["y"] // 160
    return col, row


# ---------------------------------------------------------------------------
# Level normalization + ASCII grid construction
# ---------------------------------------------------------------------------
_SLOPE_NAMES = frozenset({"Slight Slope", "Steep Slope"})


def normalize_level(lvl):
    if lvl.get("_normalized"):
        return
    lvl["_normalized"] = True

    objects = lvl.get("objects", [])

    for g in lvl.get("ground", []):
        objects.append({
            "name":"Ground","x":g["x"]*160,"y":g["y"]*160,"w":1,"h":1
        })

    if "is_overworld" in lvl:
        is_overworld = bool(lvl["is_overworld"])
    else:
        src_name = os.path.basename(lvl.get("_source_file","")).lower()
        is_overworld = "_subworld" not in src_name

    if not is_overworld:
        lvl["objects"] = objects
        return

    start_y = lvl.get("start_y",0)

    for col in range(0,7):
        for row in range(0,start_y):
            objects.append({
                "name":"Ground",
                "x":col*160,
                "y":row*160,
                "w":1,
                "h":1
            })

    goal_x = lvl.get("goal_x",0)
    goal_y = lvl.get("goal_y",0)
    goal_col = goal_x // 10

    is_castle = (lvl.get("theme_raw",-1)==2 or lvl.get("theme","")=="Castle")
    is_3dw = (lvl.get("gamestyle","")=="SM3DW" or lvl.get("gamestyle_raw",0)==22323)

    if is_castle and not is_3dw:
        objects.append({
            "name":"Goal","x":goal_col*160,"y":goal_y*160,"w":2,"h":4
        })
    else:
        objects.append({
            "name":"Goal","x":goal_col*160,"y":goal_y*160,"w":1,"h":11
        })

    for col in range(goal_col, goal_col+13):
        for row in range(0, goal_y):
            objects.append({
                "name":"Ground",
                "x":col*160,
                "y":row*160,
                "w":1,
                "h":1
            })

    lvl["objects"] = objects


def grid_bounds(lvl):
    top_b = lvl.get("top_boundary",0)
    right_b = lvl.get("right_boundary",0)
    if top_b > 0 and right_b > 0:
        max_tx = right_b // 16
        max_ty = top_b // 16
    else:
        max_tx,max_ty = 40,20
        for o in lvl.get("objects",[]):
            col,row = obj_anchor(o)
            w,h = obj_tile_size(o)
            max_tx = max(max_tx, col+w+1)
            max_ty = max(max_ty, row+h+1)
    return min(max_tx,240), min(max_ty,28)


def build_ascii_grid(level):
    objects = level.get("objects", [])
    max_tx,max_ty = grid_bounds(level)

    grid = [[" "] * max_tx for _ in range(max_ty)]

    def set_cell(col,row_game,ch):
        if 0 <= col < max_tx and 0 <= row_game < max_ty:
            grid[max_ty - 1 - row_game][col] = ch

    BG_TYPES = {"Semisolid Platform","Mushroom Platform"}
    # Terrain goes down first so ground can't cut holes in objects sitting on it.
    TERRAIN_TYPES = {"Ground","Goal"}

    # Smallest first in each layer so a 1x1 can't punch a hole in something bigger.
    order = sorted(objects, key=lambda o: obj_tile_size(o)[0] * obj_tile_size(o)[1])

    for pass_n in range(3):
        for obj in order:
            obj_name = obj.get("name","_unknown")

            # No glyph -> dropped (an empty string would misalign the row).
            if obj_name in ASCII_DROP:
                continue

            if obj_name in BG_TYPES:
                layer = 0
            elif obj_name in TERRAIN_TYPES:
                layer = 1
            else:
                layer = 2
            if layer != pass_n:
                continue

            char,_,_ = get_meta(resolve_obj_name(obj_name, level.get("gamestyle_raw", 0)))

            if obj_name in ASCII_REPLACEMENTS:
                char = ASCII_REPLACEMENTS[obj_name]

            if obj_name in _SLOPE_NAMES:
                # Slopes become a ground staircase (mirrors swe.slope_fill_cells).
                col, row = obj_anchor(obj)
                w, h = obj_tile_size(obj)
                step = 2 if obj.get("id") == 87 else 1
                descending = (obj.get("flag", 0) & 0x100000) != 0
                for x in range(w):
                    run = (w - x) if descending else (x + 1)
                    height = min((run + step - 1) // step, h)
                    for y in range(height):
                        set_cell(col + x, row + y, GROUND_CHAR)
            elif obj_name == "Mushroom Platform":
                col,row = obj_anchor(obj)
                w,h = obj_tile_size(obj)
                sc = col + w // 2
                # stem: centered column, all rows below cap
                for dy in range(h - 1):
                    set_cell(sc, row + dy, char)
                # cap: full width at top row
                for dx in range(w):
                    set_cell(col + dx, row + h - 1, char)
            elif obj_name == "Bridge":
                col,row = obj_anchor(obj)
                w,_ = obj_tile_size(obj)
                # Keep only the walkable bottom row.
                for dx in range(w):
                    set_cell(col+dx,row,char)
            elif obj_name == "Big Coin":
                # Big Coin reads as a 2x2 cluster of coins.
                col,row = obj_anchor(obj)
                coin_char,_,_ = get_meta("Coin")
                for dx in range(2):
                    for dy in range(2):
                        set_cell(col+dx,row+dy,coin_char)
            else:
                col,row = obj_anchor(obj)
                w,h = obj_tile_size(obj)

                for dx in range(w):
                    for dy in range(h):
                        set_cell(col+dx,row+dy,char)

    # Put each block's item (cid) one tile above it if that cell is empty.
    gamestyle_raw = level.get("gamestyle_raw", 0)
    for obj in objects:
        if obj.get("id") not in CONTAINER_BLOCK_IDS:
            continue
        glyph = contained_item_glyph(obj.get("cid", -1), gamestyle_raw)
        if glyph is None:
            continue
        col, row = obj_anchor(obj)
        w, h = obj_tile_size(obj)
        above = row + h               # one tile above the block's top edge
        target_col = col + w // 2     # centre column (col itself for a 1x1 block)
        if 0 <= target_col < max_tx and 0 <= above < max_ty \
                and grid[max_ty - 1 - above][target_col] == " ":
            set_cell(target_col, above, glyph)

    return ["".join(r).rstrip() for r in grid]


def level_metadata(lvl):
    """Human-readable fields to carry into the dataset, plus object boxes for the
    dataset builder. tags/difficulty were folded in earlier by
    mm2pipeline_data.toost."""
    return {
        "level_name": lvl.get("name", ""),
        "difficulty": lvl.get("difficulty"),
        "gamestyle": lvl.get("gamestyle"),
        "theme": lvl.get("theme"),
        "tags": lvl.get("tags", []),
        # No tiles for liquid, so metadata is how it travels
        "liquid_start_height": lvl.get("liquid_start_height", 0),
        "liquid_end_height": lvl.get("liquid_end_height", 0),
        "liquid_speed_raw": lvl.get("liquid_speed_raw", 0),
        "indivisible_objects": indivisible_object_boxes(lvl),
    }


def json_to_ascii_file(infile, outdir, metadata=None):
    data = json.loads(Path(infile).read_text(encoding="utf-8"))
    levels = data if isinstance(data, list) else [data]

    for lvl in levels:
        lvl.setdefault("_source_file", str(infile))
        normalize_level(lvl)

    for idx,lvl in enumerate(levels, start=1):
        stem = Path(infile).stem
        suffix = f"_{idx}" if len(levels) > 1 else ""
        out_stem = f"{stem}{suffix}"
        outfile = Path(outdir) / f"{out_stem}.txt"
        outfile.write_text("\n".join(build_ascii_grid(lvl)) + "\n", encoding="utf-8")
        # Keyed by ascii stem so mm2pipeline_data.dataset can match it back.
        if metadata is not None:
            metadata[out_stem] = level_metadata(lvl)


def main_json_to_ascii(argv=None):
    ap = argparse.ArgumentParser(description="Convert MM2 level JSON to ASCII grids.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--metadata_output", default=None,
                    help="Where to write the per-level metadata JSON (level_name, "
                         "difficulty, gamestyle, theme, tags), keyed by ascii file "
                         "stem. Default: <output_folder>/metadata.json. This is the "
                         "file mm2pipeline_data.dataset reads with --metadata.")
    args = ap.parse_args(argv)

    outdir = Path(args.output_folder)
    outdir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    for jf in sorted(Path(args.input).glob("*.json")):
        try:
            json_to_ascii_file(jf, outdir, metadata)
            print(f"Converted {jf.name}")
        except Exception as e:
            print(f"Failed {jf.name}: {e}")

    meta_path = Path(args.metadata_output) if args.metadata_output else outdir / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote metadata for {len(metadata)} level(s) -> {meta_path}")


# ===========================================================================
# Reverse: ASCII -> JSON
# ===========================================================================

TILE = 160          # JSON sub-pixel units per tile (160 = 1 tile)
TILE_CENTER = 80    # real .bcd objects anchor at the tile CENTER (col*160 + 80)
GROUND_TILE_PX = 16  # boundary fields are in pixels (16 px / tile), per toost

# Baseline flag for reconstructed objects; modifiers can't be recovered from ASCII.
DEFAULT_FLAG = 0x6000040


def parse_ascii(text):
    """Return (rows, width) with rows[0] as game row 0 (bottom of the level)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":     # drop the final "\n"; keep interior blanks
        lines.pop()
    width = max((len(l) for l in lines), default=0)
    rows = list(reversed(lines))      # file is top-to-bottom; index from bottom
    return rows, width


# DEFAULT_FLAG already carries 0x40 (mouth-up); a horizontal pipe clears it.
PIPE_FLAG_U = DEFAULT_FLAG
PIPE_FLAG_R = DEFAULT_FLAG & ~0x60


def make_object(name, col, row, w=1, h=1, flag=DEFAULT_FLAG):
    """Build a base-schema w x h object with bottom-left tile at (col, row).
    Left-anchored objects (_REV_LEFT_ANCHOR) store x = col*160 + 80; everything
    else stores the span centre. y is the bottom-row centre."""
    if name in _REV_LEFT_ANCHOR:
        x = col * TILE + TILE_CENTER
    else:
        x = (col + w // 2) * TILE + (TILE_CENTER if w % 2 else 0)
    return {
        "name": name,
        "x": x,
        "y": row * TILE + TILE_CENTER,
        "w": w,
        "h": h,
        "flag": flag,
        "cflag": flag,
        "ex": 0,
        "id": NAME_TO_ID[name],
        "cid": -1,
        "lid": -1,
        "sid": -1,
        "link_type": 0,
    }


# ---------------------------------------------------------------------------
# Multi-tile object coalescing (ASCII -> JSON)
#
# build_ascii_grid paints each object as a w x h block of its glyph, so coalesce()
# groups touching same-glyph cells back into one object. Anything not listed
# stays 1x1 so rows of coins don't get merged.
# ---------------------------------------------------------------------------
_FIXED, _BBOX, _MUSHROOM, _HRUN, _VRUN, _PIPE = (
    "fixed", "bbox", "mushroom", "hrun", "vrun", "pipe")

# Objects whose x is the LEFT-tile centre (col = x // 160).
_REV_LEFT_ANCHOR = frozenset({
    "Mushroom Platform", "Semisolid Platform", "Bridge", "Pipe",
    "One-Way Wall", "One-Way",
})

# name -> policy. "confirmed" means the size was seen in a real toost export.
COALESCE_POLICY = {
    "Thwomp":             (_FIXED, 2, 2),   # confirmed
    "Bowser":             (_FIXED, 2, 2),   # confirmed
    "Big Coin":           (_FIXED, 2, 2),   # confirmed
    "Checkpoint Flag":    (_FIXED, 2, 2),   # confirmed
    "Goomba's Shoe":      (_FIXED, 2, 2),   # confirmed (decoder: "Shoe Goomba")
    "Yoshi's Egg":        (_FIXED, 2, 2),   # id 45, SMW/NSMBU form of the above
    "Saw":                (_FIXED, 3, 3),   # confirmed 3x3
    "Swinging Claw":      (_FIXED, 3, 4),   # confirmed
    "Skewer":             (_FIXED, 4, 4),
    "Donut":              (_FIXED, 3, 3),   # id 82 Donut Block Platform
    "Boom Boom":          (_FIXED, 2, 2),   # confirmed
    "Banzai Bill":        (_FIXED, 4, 4),   # confirmed, never smaller
    "Angry Sun":          (_FIXED, 2, 2),   # confirmed
    "Clown Car":          (_FIXED, 2, 2),   # confirmed
    "Door":               (_FIXED, 1, 2),   # confirmed
    "Lakitu":             (_FIXED, 1, 2),   # confirmed
    # Wiggler, Chain Chomp and Bowser Jr. are left out since they're 1x1 in the data.
    "Mushroom Platform":  (_MUSHROOM,),
    "Semisolid Platform": (_BBOX,),   # box repair, see repair_semisolid_cells
    "Half-Collision Platform": (_BBOX,),
    "One-Way Wall":       (_BBOX,),
    "One-Way":            (_BBOX,),
    "Bridge":             (_HRUN,),
    "Lift":               (_HRUN,),
    "Bullet Bill Blaster":(_VRUN,),
    "Vine":               (_VRUN,),
    "Goal":               (_VRUN,),   # one tall pole, not a stack of tiles
    "Pipe":               (_PIPE,),
}

# Objects with a fixed footprint. A window must never cut one of these in half.
INDIVISIBLE_NAMES = frozenset(
    name for name, policy in COALESCE_POLICY.items() if policy[0] == _FIXED)


def indivisible_object_boxes(level):
    """Boxes of the objects a cut must not land inside, as {name,x,y,w,h}. Grid
    coordinates, y counted from the top like the emitted rows."""
    _, max_ty = grid_bounds(level)
    gamestyle_raw = level.get("gamestyle_raw", 0)

    boxes = []
    for obj in level.get("objects", []):
        name = obj.get("name", "_unknown")
        if name in ASCII_DROP:
            continue

        resolved = resolve_obj_name(name, gamestyle_raw)
        if resolved not in INDIVISIBLE_NAMES:
            continue
        col, row_game = obj_anchor(obj)
        w, h = obj_tile_size(obj)
        if w <= 1 and h <= 1:   # a single cell survives any cut
            continue
        boxes.append({"name": resolved, "x": col, "y": max_ty - row_game - h,
                      "w": w, "h": h})
    return boxes


def _connected_components(cells):
    """4-connected components of a set of (col, row) cells."""
    cells = set(cells)
    seen = set()
    comps = []
    for start in cells:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            c, r = stack.pop()
            comp.append((c, r))
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (c + dc, r + dr)
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def _runs(values):
    """Yield (start, length) for each maximal run of consecutive ints."""
    values = sorted(values)
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
        else:
            yield start, prev - start + 1
            start = prev = v
    yield start, prev - start + 1


def _pipe_split(comp, vertical):
    """Pair a pipe columns (rows when not vertical) off into 2-wide pipes since a vert/horizantal pipe will always be either 2 * h or w * 2."""
    lanes = {}
    for c, r in comp:
        if vertical:
            lanes.setdefault(c, set()).add(r)
        else:
            lanes.setdefault(r, set()).add(c)
    lo, hi = min(lanes), max(lanes)
    if (hi - lo) % 2 == 0:
        return None
    pipes = []
    for a in range(lo, hi + 1, 2):
        if lanes[a] != lanes[a + 1]:
            return None
        for start, length in _runs(lanes[a]):
            if length < 2:
                return None
            pipes.append((a, start, length))
    return pipes


# ---------------------------------------------------------------------------
# Semisolid platform repair (ASCII -> JSON)
#
# The model sprays 'k' as lone tiles and ragged blobs, so each blob gets rebuilt
# as one box, widened to 3 tiles and dragged down to the ground.
# ---------------------------------------------------------------------------
PLATFORM_MIN_SIZE = 3       # SMM2 won't place a semisolid smaller than 3x3


def repair_semisolid_cells(cells, ground=None):
    """Turn semisolid 'k' cells into (col, row, w, h) boxes, one per 4-connected
    blob, widened to the 3-tile minimum and dragged down to `ground` (set of
    solid cells; falsy = drag to the floor)."""
    ground = ground or set()
    boxes = []
    for comp in _connected_components(cells):
        cols = [c for c, _ in comp]
        rows = [r for _, r in comp]
        c0, c1 = min(cols), max(cols)
        cap_row = max(rows)                        # top / walkable surface
        w = c1 - c0 + 1
        if w < PLATFORM_MIN_SIZE:                  # widen symmetrically
            c0 = max(0, c0 - (PLATFORM_MIN_SIZE - w) // 2)
            w = PLATFORM_MIN_SIZE
        c1 = c0 + w - 1
        # Same rest-on-ground rule as json_to_bcd._fix_platform_objects.
        bottom = min(rows)
        while bottom > 0 and any((c, bottom - 1) not in ground
                                 for c in range(c0, c1 + 1)):
            bottom -= 1
        if cap_row - bottom + 1 < PLATFORM_MIN_SIZE:   # 3-tall minimum
            bottom = max(0, cap_row - (PLATFORM_MIN_SIZE - 1))
        boxes.append((c0, bottom, w, cap_row - bottom + 1))
    return boxes


def coalesce(name, cells, out, ground=None):
    """Turn `cells` (the glyph cells for `name`) into objects per COALESCE_POLICY,
    appending to `out`. `ground` drags semisolids to the floor. No policy = 1x1."""
    policy = COALESCE_POLICY.get(name)
    if policy is None:
        for col, row in cells:
            out.append(make_object(name, col, row))
        return

    kind = policy[0]
    if name == "Semisolid Platform":
        for col, row, w, h in repair_semisolid_cells(cells, ground):
            out.append(make_object(name, col, row, w, h))
        return
    for comp in _connected_components(cells):
        cols = [c for c, _ in comp]
        rows = [r for _, r in comp]
        c0, c1, r0, r1 = min(cols), max(cols), min(rows), max(rows)

        if kind == _FIXED:
            fw, fh = policy[1], policy[2]
            # A full block twice the footprint is one big enemy, not four.
            if (c1 - c0 + 1, r1 - r0 + 1) == (2 * fw, 2 * fh) and len(comp) == 4 * fw * fh:
                out.append(make_object(name, c0, r0, 2 * fw, 2 * fh))
                continue
            cr = r0
            while cr <= r1:                # cr is each stamp's bottom row
                cc = c0
                while cc <= c1:
                    out.append(make_object(
                        name, cc, cr,
                        min(fw, c1 - cc + 1), min(fh, r1 - cr + 1)))
                    cc += fw
                cr += fh
        elif kind == _BBOX:
            out.append(make_object(name, c0, r0, c1 - c0 + 1, r1 - r0 + 1))
        elif kind == _MUSHROOM:
            # Stacked mushrooms share a blob, so each cap hangs down to the next one.
            rows_by = {}
            for c, r in comp:
                rows_by.setdefault(r, []).append(c)
            cap_rows = sorted(
                (r for r, cs in rows_by.items()
                 if len(cs) >= 2 and len(rows_by.get(r + 1, ())) <= 1),
                reverse=True)
            if not cap_rows:                       # stem-only speck: treat top
                cap_rows = [r1]
            for i, rc in enumerate(cap_rows):
                cs = rows_by[rc]
                cl = min(cs)
                bottom = cap_rows[i + 1] + 1 if i + 1 < len(cap_rows) else r0
                out.append(make_object(name, cl, bottom,
                                       max(cs) - cl + 1, rc - bottom + 1))
        elif kind == _HRUN:
            by_row = {}
            for c, r in comp:
                by_row.setdefault(r, []).append(c)
            for r, rcols in by_row.items():
                for start, length in _runs(rcols):
                    out.append(make_object(name, start, r, length, 1))
        elif kind == _VRUN:
            by_col = {}
            for c, r in comp:
                by_col.setdefault(c, []).append(r)
            for c, crows in by_col.items():
                for start, length in _runs(crows):
                    out.append(make_object(name, c, start, 1, length))
        elif kind == _PIPE:
            # Pipes standing side by side share a dimension, so it gets split into 2 wide pipes, where we choose to have the way with fewer pipes always. Should work with vertical pipes.
            up = _pipe_split(comp, True)
            side = _pipe_split(comp, False)
            if side is not None and (up is None or len(side) < len(up)):
                for r, c, length in side:       # horizontal -> mouth right
                    out.append(make_object(name, c, r + 1, 2, length, PIPE_FLAG_R))
            elif up is not None:                # vertical -> mouth up (default)
                for c, r, length in up:
                    out.append(make_object(name, c, r, 2, length, PIPE_FLAG_U))
            else:                          # malformed -> leave as 1x1 cells
                for col, row in comp:
                    out.append(make_object(name, col, row))


# ---------------------------------------------------------------------------
# End-of-level goal synthesis (ASCII -> JSON)
#
# Training data has no 'G', so generated levels get a runway and goal added on
# the right edge.
# ---------------------------------------------------------------------------
GOAL_RUNWAY_TILES = 10      # width of the synthesized end platform, in tiles
GOAL_WIDTH_TILES = 3        # the goal's own footprint, in tiles


def _append_end_goal(ground, width):
    """Append a ground runway past the right edge and return
    (goal_col, goal_row, added_cols) for a goal standing on its first tile."""
    ground_at = {(g["x"], g["y"]) for g in ground}
    # Runway matches the floor height at the right-most column, or 2 if there isn't one.
    floor = 0
    for col in range(width - 1, -1, -1):
        if (col, 0) in ground_at:
            while (col, floor) in ground_at:
                floor += 1
            break
    if floor == 0:
        floor = 2
    runway_left = width                     # append just past the current content
    runway = max(GOAL_RUNWAY_TILES, GOAL_WIDTH_TILES)   # must hold the goal
    for col in range(runway_left, runway_left + runway):
        for row in range(floor):
            ground.append({"x": col, "y": row, "id": 0, "bid": 0})
    return runway_left, floor, runway


LIQUID_MODES = {0: "Static", 1: "Rising or Falling", 2: "Rising and Falling"}
LIQUID_SPEEDS = {0: "None", 1: "x1", 2: "x2", 3: "x3"}


def ascii_to_level(text, source_file=None, *, gamestyle_raw=22349, theme_raw=0,
                   timer=300, liquid_start_height=0, liquid_end_height=0,
                   liquid_mode_raw=0, liquid_speed_raw=0):
    rows, width = parse_ascii(text)
    height = len(rows)

    ground = []
    objects = []
    unknown_glyphs = {}
    goal_cells = []
    # name -> set of (col, row) cells, coalesced into multi-tile objects below.
    obj_cells = {}

    for row_game, line in enumerate(rows):
        for col, ch in enumerate(line):
            if ch == " ":
                continue
            if ch == GROUND_CHAR:
                # Autotile id/bid aren't recoverable; toost re-derives them.
                ground.append({"x": col, "y": row_game, "id": 0, "bid": 0})
                continue
            name = CHAR_TO_NAME.get(ch)
            if name is None or name not in NAME_TO_ID:
                unknown_glyphs[ch] = unknown_glyphs.get(ch, 0) + 1
                continue
            if name == "Goal":
                # The goal lives in goal_x/goal_y, not objects[], so just collect it.
                goal_cells.append((col, row_game))
                continue
            obj_cells.setdefault(name, set()).add((col, row_game))

    # An item right above a Brick/?/Hidden block goes back into that block's cid.
    block_cells = set()
    for bname in CONTAINER_BLOCK_NAMES:
        block_cells |= obj_cells.get(bname, set())
    block_item = {}  # (col, row) of a block -> contained item's MM2 object id
    for iname in list(obj_cells):
        meta = OBJ_META.get(iname)
        if not meta or meta[2] != CAT_ITEM:
            continue
        remaining = set()
        for cell in obj_cells[iname]:
            below = (cell[0], cell[1] - 1)
            if below in block_cells and below not in block_item:
                block_item[below] = NAME_TO_ID[iname]
            else:
                remaining.add(cell)
        obj_cells[iname] = remaining

    # Coalesce same-glyph cells into correctly-sized objects (see coalesce()).
    ground_cells = {(g["x"], g["y"]) for g in ground}
    for name, cells in obj_cells.items():
        if name in CONTAINER_BLOCK_NAMES:
            # Emitted here (one per cell) so the recovered cid can be attached.
            for col, row in cells:
                obj = make_object(name, col, row)
                cid = block_item.get((col, row))
                if cid is not None:
                    obj["cid"] = cid
                objects.append(obj)
            continue
        coalesce(name, cells, objects, ground_cells)

    # goal_x is in tenths of a tile. No 'G' means a goal gets added on a new runway.
    if goal_cells:
        goal_col = min(c for c, _ in goal_cells)
        goal_row = min(r for _, r in goal_cells)
    else:
        goal_col, goal_row, added = _append_end_goal(ground, width)
        width += added

    # Spawn on the ground at the left edge, since start_y=0 spawns in the void.
    ground_at = {(g["x"], g["y"]) for g in ground}
    start_y = 0
    for c in (0, 1, 2):
        h = 0
        while (c, h) in ground_at:
            h += 1
        if h:
            start_y = h
            break
    if start_y == 0:
        start_y = 2

    stem = Path(source_file).stem if source_file else "level"

    level = {
        "name": stem,
        "description": "Reconstructed from ASCII by mm2pipeline_data.ascii",
        "gamestyle": GAMESTYLE_NAME.get(gamestyle_raw, "SMW"),
        "gamestyle_raw": gamestyle_raw,
        "theme": THEME_NAME.get(theme_raw, "Ground"),
        "theme_raw": theme_raw,
        # False suppresses normalize_level()'s synthetic start/goal ground.
        "is_overworld": False,
        "night_time": False,
        "clear_time": 0,
        "clear_attempts": 0,
        "game_version": "0.0.0",
        "game_version_raw": 0,
        "timer": timer,
        "start_y": start_y,
        "goal_x": goal_col * 10,
        "goal_y": goal_row,
        "clear_condition_type": "None",
        "clear_condition_type_raw": 0,
        "clear_condition_magnitude": 0,
        "clear_condition": "None",
        "clear_condition_category": "None",
        "clear_condition_category_raw": 0,
        "autoscroll_speed": "x1",
        "autoscroll_speed_raw": 0,
        "autoscroll_type": "None",
        "autoscroll_type_raw": 0,
        "orientation": "Horizontal",
        "orientation_raw": 0,
        "liquid_start_height": liquid_start_height,
        "liquid_end_height": liquid_end_height,
        "liquid_mode": LIQUID_MODES.get(liquid_mode_raw, "Static"),
        "liquid_speed": LIQUID_SPEEDS.get(liquid_speed_raw, "None"),
        "boundary_type": "Built Above Line",
        "liquid_mode_raw": liquid_mode_raw,
        "liquid_speed_raw": liquid_speed_raw,
        "boundary_type_raw": 0,
        # Boundaries are in pixels (16 px / tile), per toost.
        "right_boundary": width * GROUND_TILE_PX,
        "top_boundary": height * GROUND_TILE_PX,
        "left_boundary": 0,
        "bottom_boundary": 0,
        "object_count": len(objects),
        "ground_count": len(ground),
        "upload_id": 0,
        "creation_id": 0,
        "gamever": 0,
        "management_flags": 0,
        "objects": objects,
        "ground": ground,
        "track": [],
        "clear_pipes": [],
        "snakes": [],
        "piranha_creepers": [],
    }
    if source_file is not None:
        level["_source_file"] = str(source_file)
    if unknown_glyphs:
        level["_unknown_glyphs"] = unknown_glyphs
    return level


def ascii_to_json_file(infile, outdir, **kwargs):
    # utf-8-sig transparently strips a leading BOM if one is present.
    text = Path(infile).read_text(encoding="utf-8-sig")
    level = ascii_to_level(text, source_file=infile, **kwargs)
    # The pipeline matches files by the _overworld/_subworld suffix.
    stem = Path(infile).stem
    if not (stem.endswith("_overworld") or stem.endswith("_subworld")):
        stem += "_overworld"
    outfile = Path(outdir) / f"{stem}.json"
    outfile.write_text(json.dumps(level, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    if level.get("_unknown_glyphs"):
        print(f"  warning: unmapped glyphs in {Path(infile).name}: "
              f"{level['_unknown_glyphs']}")


def main_ascii_to_json(argv=None):
    ap = argparse.ArgumentParser(description="Convert ASCII Mario Maker grids back to MM2 JSON.")
    ap.add_argument("--input", required=True, help="folder of .txt files, or a single .txt file")
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--gamestyle", choices=sorted(GAMESTYLE_RAW), default="smw",
                    help="game style for the rebuilt level (default: smw)")
    ap.add_argument("--theme", choices=sorted(THEME_RAW), default="overworld",
                    help="course theme (default: overworld)")
    ap.add_argument("--timer", type=int, default=300, help="level timer (default: 300)")
    args = ap.parse_args(argv)

    outdir = Path(args.output_folder)
    outdir.mkdir(parents=True, exist_ok=True)

    inpath = Path(args.input)
    files = [inpath] if inpath.is_file() else sorted(inpath.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {args.input}")
        return

    kwargs = dict(
        gamestyle_raw=GAMESTYLE_RAW[args.gamestyle],
        theme_raw=THEME_RAW[args.theme],
        timer=args.timer,
    )
    for tf in files:
        try:
            ascii_to_json_file(tf, outdir, **kwargs)
            print(f"Converted {tf.name}")
        except Exception as e:
            print(f"Failed {tf.name}: {e}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] not in ("to-ascii", "to-json"):
        print("Usage: python -m mm2pipeline_data.ascii {to-ascii|to-json} [options]")
        print("  to-ascii   level JSON folder -> ASCII grids (+ metadata.json)")
        print("  to-json    ASCII grids -> level JSON")
        sys.exit(2)
    if argv[0] == "to-ascii":
        main_json_to_ascii(argv[1:])
    else:
        main_ascii_to_json(argv[1:])


if __name__ == "__main__":
    main()
