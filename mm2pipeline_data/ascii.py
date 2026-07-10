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
def obj_tile_size(obj: dict):
    """(w, h) in tiles. Pipes use h as length regardless of direction; the
    cross-section is always 2."""
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

    for pass_n in range(2):
        for obj in objects:
            obj_name = obj.get("name","_unknown")

            # No glyph -> dropped (an empty string would misalign the row).
            if obj_name in ASCII_DROP:
                continue

            is_bg = obj_name in BG_TYPES
            if pass_n == 0 and not is_bg:
                continue
            if pass_n == 1 and is_bg:
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

    # Surface each block's contained item (cid) as its glyph one tile above
    # the block, only into a still-empty cell.
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
    """Human-readable fields to carry into the dataset. tags/difficulty were
    folded in earlier by mm2pipeline_data.toost."""
    return {
        "level_name": lvl.get("name", ""),
        "difficulty": lvl.get("difficulty"),
        "gamestyle": lvl.get("gamestyle"),
        "theme": lvl.get("theme"),
        "tags": lvl.get("tags", []),
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
# build_ascii_grid paints each object as a w x h block of its glyph. Reading it
# back one cell at a time would make a 2x2 Thwomp four thwomps, an N-wide
# platform N tiny ones, and so on. coalesce() regroups 4-connected same-glyph
# cells into correctly-sized objects. Footprints below come from real toost JSON
# exports; a few bosses with no sample are assumed 2x2 (harmless -- they're never
# placed adjacent, and an isolated cell clamps to one object). Objects not listed
# stay 1x1, so rows of spikes/coins are never merged.
# ---------------------------------------------------------------------------
_FIXED, _BBOX, _MUSHROOM, _HRUN, _VRUN, _PIPE = (
    "fixed", "bbox", "mushroom", "hrun", "vrun", "pipe")

# Objects whose x is the LEFT-tile centre (col = x // 160).
_REV_LEFT_ANCHOR = frozenset({
    "Mushroom Platform", "Semisolid Platform", "Bridge", "Pipe",
    "One-Way Wall", "One-Way",
})

# name -> policy. _FIXED tiles into fw x fh stamps, _BBOX one box per component,
# _MUSHROOM a cap+stem platform, _HRUN/_VRUN 1-tall/1-wide runs, _PIPE a 2-wide
# directional pipe. "confirmed" = seen in a real toost export.
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
    "Boom Boom":          (_FIXED, 2, 2),   # assumed
    "Banzai Bill":        (_FIXED, 2, 2),   # assumed
    "Angry Sun":          (_FIXED, 2, 2),   # assumed
    "Bowser Jr.":         (_FIXED, 2, 2),   # assumed
    "Bowser Jr":          (_FIXED, 2, 2),
    "Clown Car":          (_FIXED, 2, 2),   # assumed
    "Door":               (_FIXED, 1, 2),   # pairing the halves stops mispairing
    # Wiggler/Chain Chomp deliberately absent: 1x1 in real data and often in rows.
    "Mushroom Platform":  (_MUSHROOM,),
    "Semisolid Platform": (_BBOX,),   # box repair, see repair_semisolid_cells
    "Half-Collision Platform": (_BBOX,),
    "One-Way Wall":       (_BBOX,),
    "One-Way":            (_BBOX,),
    "Bridge":             (_HRUN,),
    "Lift":               (_HRUN,),
    "Bullet Bill Blaster":(_VRUN,),
    "Vine":               (_VRUN,),
    "Pipe":               (_PIPE,),
}


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


# ---------------------------------------------------------------------------
# Semisolid platform repair (ASCII -> JSON)
#
# A semisolid is a solid box you stand on and pass through from below. The model
# sprays its 'k' glyph as lone tiles and ragged blobs, so plain _BBOX leaves a
# swarm of slivers. repair_semisolid_cells() rebuilds one box per 4-connected
# blob: take its columns and top (cap) row, widen to the 3-tile minimum, then
# drag the bottom down to rest on ground.
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
        # Drag the bottom down until every column directly beneath the base rests
        # on ground (or we hit the floor) -- the same rest-on-ground rule
        # json_to_bcd._fix_platform_objects uses.
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
            # A mushroom is a wide cap over a 1-wide stem; stacked mushrooms share
            # a blob. A cap is a row of >=2 cells with a narrow/empty row above;
            # each cap hangs down to just above the next cap (or the blob bottom).
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
            bw, bh = c1 - c0 + 1, r1 - r0 + 1
            if bw == 2 and bh >= 2:        # vertical -> mouth up (default)
                out.append(make_object(name, c0, r0, 2, bh, PIPE_FLAG_U))
            elif bh == 2 and bw >= 2:      # horizontal -> mouth right
                out.append(make_object(name, c0, r1, 2, bw, PIPE_FLAG_R))
            else:                          # malformed -> leave as 1x1 cells
                for col, row in comp:
                    out.append(make_object(name, col, row))


# ---------------------------------------------------------------------------
# End-of-level goal synthesis (ASCII -> JSON)
#
# With --strip_goal training data the model produces levels with no 'G'. This
# tacks a reachable finish onto the right edge: a flat ground runway flush with
# the level's floor, with a goal standing on its first tile.
# ---------------------------------------------------------------------------
GOAL_RUNWAY_TILES = 10      # width of the synthesized end platform, in tiles
GOAL_WIDTH_TILES = 3        # the goal's own footprint, in tiles


def _append_end_goal(ground, width):
    """Append a ground runway past the right edge and return
    (goal_col, goal_row, added_cols) for a goal standing on its first tile."""
    ground_at = {(g["x"], g["y"]) for g in ground}
    # Floor height = the ground stack at the right-most column that has a floor;
    # the runway is laid flush with it. Default to SMM2's usual floor of 2.
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


def ascii_to_level(text, source_file=None, *, gamestyle_raw=22349, theme_raw=0,
                   timer=300):
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
                # The goal is header metadata (goal_x/goal_y), not an objects[]
                # entry, so collect its cells and recover the pole base below.
                goal_cells.append((col, row_game))
                continue
            obj_cells.setdefault(name, set()).add((col, row_game))

    # Fold an item glyph directly above a Brick/?/Hidden block into that block's
    # `cid` (inverse of the forward painter), so it round-trips as a container.
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

    # Recover goal_x/goal_y from the flagpole cells (goal_x in tenths of a tile,
    # goal_y the base row). The goal is left-anchored and grows up, so take the
    # min column/row. With no 'G' glyphs, synthesize a goal on a new runway.
    if goal_cells:
        goal_col = min(c for c, _ in goal_cells)
        goal_row = min(r for _, r in goal_cells)
    else:
        goal_col, goal_row, added = _append_end_goal(ground, width)
        width += added

    # Recover the spawn height from the ground stack at the left edge (start_y=0
    # would spawn the player in the void). Default to SMM2's usual 2.
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
        "liquid_start_height": 0,
        "liquid_end_height": 0,
        "liquid_mode": "None",
        "liquid_speed": "x1",
        "boundary_type": "Built Above Line",
        "liquid_mode_raw": 0,
        "liquid_speed_raw": 0,
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
    # The pipeline finds companions by the _overworld/_subworld suffix, so tag
    # generated levels as overworld.
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
