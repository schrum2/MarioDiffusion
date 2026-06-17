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


def parse_mmlv(path: Path) -> Dict[Tuple[int,int], dict]:
    """Return sparse dict of (tile_x, tile_y) -> {field: float_value}."""
    text = path.read_bytes().decode("utf-8", errors="replace")
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
        if d == 4.0:   return "a"   # cannon / turret enemy
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


def mmlv_to_grid(path: Path):
    """Convert one .mmlv to a 2-D list of VGLC chars, cropped to content."""
    cells = parse_mmlv(path)

    # Collect only cells that produce a character
    char_cells: Dict[Tuple[int,int], str] = {}
    for coord, cell in cells.items():
        ch = classify(cell)
        if ch is not None:
            char_cells[coord] = ch

    # Also need player spawn – look for the [Player] or spawn marker.
    # In .mmlv the player start is stored under a separate section as
    # plain "x=... y=..." lines (not the tile grid). Parse it separately.
    text = path.read_bytes().decode("utf-8", errors="replace")
    spawn_match = re.search(
        r'\[(?:Player|PlayerData)\].*?x=(\d+).*?y=(\d+)', text,
        re.DOTALL | re.IGNORECASE
    )
    if spawn_match:
        sx = int(spawn_match.group(1)) // TILE_PX
        sy = int(spawn_match.group(2)) // TILE_PX
        char_cells[(sx, sy)] = "P"

    if not char_cells:
        return []  # empty level

    # Bounding box of content
    xs = [c[0] for c in char_cells]
    ys = [c[1] for c in char_cells]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width  = max_x - min_x + 1
    height = max_y - min_y + 1

    # Build grid (empty = '-')
    grid = [['-'] * width for _ in range(height)]
    for (tx, ty), ch in char_cells.items():
        row = ty - min_y
        col = tx - min_x
        grid[row][col] = ch

    # Strip rows that are entirely '-' from top and bottom
    def is_empty_row(r):
        return all(c == '-' for c in r)

    while grid and is_empty_row(grid[0]):
        grid.pop(0)
    while grid and is_empty_row(grid[-1]):
        grid.pop()

    if not grid:
        return grid

    # Strip empty columns from left and right
    def is_empty_col(grid, col):
        return all(row[col] == '-' for row in grid)

    left  = 0
    right = len(grid[0]) - 1
    while left <= right and is_empty_col(grid, left):
        left += 1
    while right >= left and is_empty_col(grid, right):
        right -= 1

    if left > 0 or right < len(grid[0]) - 1:
        grid = [row[left:right+1] for row in grid]

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