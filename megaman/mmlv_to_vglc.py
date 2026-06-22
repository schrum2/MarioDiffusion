"""
mmlv_to_vglc.py  –  Convert Mega Man Maker .mmlv files to VGLC ASCII .txt

Format discoveries from reverse-engineering real .mmlv files:
  - Entries look like:  fieldXpx,Ypx="value"
  - Coordinates are in pixels (multiply by 16 per tile)
  - Only occupied cells appear (sparse format)
  - Fields per cell:
      i = tile type:  1=solid block, 2=spike, 3=ladder
      d = entity type: 4=cannon(on block), 5=enemy or border(o=9999), 6=enemy, 7=item, 8=boss/orb, 10=moving platform
      e = subtype (tile variant or enemy ID — ignored for tile classification)
      o = 9999.0 means invisible border wall (skip it)
      j,k = texture IDs (ignored)
      a,b,f,h = misc flags (ignored)

VGLC character mapping (matches your MM.json tileset):
  # = solid block
  H = spike / hazard
  | = ladder (climbable)
  - = empty / passable
  @ = null (outside level bounds, trimmed later)
  P = player spawn
  Z = collectible / powerup
  a = enemy (generic)
  M = moving platform
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


def parse_mmlv(path: Path) -> Dict[Tuple[int,int], dict]:
    """Return sparse dict of (tile_x, tile_y) -> {field: float_value}."""
    text = path.read_bytes().decode("utf-8", errors="replace").replace('\r', '')
    cells: Dict[Tuple[int,int], dict] = {}
    for m in re.finditer(r'([a-z])(\d+),(\d+)="([^"]+)"', text):
        field = m.group(1)
        tx = int(m.group(2)) // TILE_PX
        ty = int(m.group(3)) // TILE_PX
        val = float(m.group(4))
        if (tx, ty) not in cells:
            cells[(tx, ty)] = {}
        cells[(tx, ty)][field] = val
    return cells


def classify(cell: dict) -> str:
    """Map a cell's fields to a VGLC character."""
    i = cell.get("i")
    d = cell.get("d")
    o = cell.get("o")

    # Skip invisible border walls
    if o == 9999.0:
        return None  # type: ignore  # caller handles None

    # Entity layer (d field) takes priority over tile layer
    if d is not None:
        if d == 4.0:
            e = cell.get("e")
            if e == 0.0 or e is None:  return "P"   # player spawn (Mega Man)
            if e == 1.0:               return "P"   # player spawn (Proto Man)
            if e == 2.0:               return "P"   # player spawn (Bass)
            return "a"                              # actual cannon/turret
        if d == 5.0:   return "a"   # enemy (various)
        if d == 6.0:   return "a"   # enemy (various)
        if d == 7.0:   return "Z"   # item / powerup
        if d == 8.0:   return "Z"   # boss orb / checkpoint
        if d == 10.0:
            # Moving platform — also has i=1, so check i first below;
            # but since d is checked first, mark as M
            return "M"
        # Unknown entity — treat as generic enemy
        return "a"

    # Tile layer (i field)
    if i == 1.0:   return "#"   # solid block
    if i == 2.0:   return "H"   # spike
    if i == 3.0:   return "|"   # ladder

    # Cell exists but no recognised i or d (metadata-only cell)
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

def mmlv_to_grid(path: Path):
    """Convert one .mmlv to a 2-D list of VGLC chars."""
    cells = parse_mmlv(path)

    char_cells: Dict[Tuple[int,int], str] = {}
    for coord, cell in cells.items():
        ch = classify(cell)
        if ch is not None:
            char_cells[coord] = ch

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

def convert(src: Path, dst: Path):
    grid = mmlv_to_grid(src)
    if not grid:
        print(f"  [skip] {src.name} — no content found")
        return False
    dst.write_text("\n".join("".join(row) for row in grid) + "\n", encoding="utf-8")
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    print(f"  {src.name} → {dst.name}  ({cols}w × {rows}h)")
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