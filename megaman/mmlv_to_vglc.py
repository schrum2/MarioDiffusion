"""Convert a Mega Man Maker .mmlv file to a VGLC-style ASCII text file.

Reads the tile entity entries from a .mmlv file and reconstructs a grid
using VGLC ASCII characters. Detail is lost (sprite IDs, exact entity
properties) but the level layout is preserved.

Mega Man Maker entity IDs -> VGLC characters:
    0   -> P  (player 1 start)
    2   -> E  (enemy)
    3   -> X  (solid block)
    4   -> p  (player 2 start)
    7   -> E  (enemy variant)
    9   -> #  (ladder)
    15  -> L  (unknown/other)
    20  -> B  (boss trigger)
    98  -> S  (spike)
    anything else -> ?
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TILE_SIZE = 16
Y_OFFSET = 64   # level content starts at pixel y=64

# Mega Man Maker entity ID -> VGLC character
ID_TO_CHAR: Dict[int, str] = {
    0:  "P",   # player 1 start
    2:  "E",   # enemy
    3:  "X",   # solid block
    4:  "p",   # player 2 start
    7:  "E",   # enemy variant
    9:  "#",   # ladder
    15: "L",
    20: "B",   # boss trigger
    98: "S",   # spike
}
UNKNOWN_CHAR = "?"


def parse_mmlv(path: Path) -> Tuple[Dict[Tuple[int, int], int], int, int]:
    """
    Parse a .mmlv file and return (tile_dict, width_in_tiles, height_in_tiles).
    tile_dict maps (col, row) -> entity_id.
    """
    tiles: Dict[Tuple[int, int], int] = {}
    # Match lines like: e<x>,<y>="<val>.000000"
    entry_re = re.compile(r'^e(-?\d+),(-?\d+)="([0-9.]+)"')

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        m = entry_re.match(line)
        if not m:
            continue
        px, py, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if py < Y_OFFSET:
            continue   # skip the top border / header area
        col = px // TILE_SIZE
        row = (py - Y_OFFSET) // TILE_SIZE
        entity_id = int(val)
        tiles[(col, row)] = entity_id

    if not tiles:
        return {}, 0, 0

    max_col = max(c for c, r in tiles)
    max_row = max(r for c, r in tiles)
    return tiles, max_col + 1, max_row + 1


def mmlv_to_vglc(path: Path) -> List[str]:
    tiles, width, height = parse_mmlv(path)
    if not tiles:
        return []

    rows: List[str] = []
    for row in range(height):
        line = ""
        for col in range(width):
            entity_id = tiles.get((col, row))
            if entity_id is None:
                line += "-"
            else:
                line += ID_TO_CHAR.get(entity_id, UNKNOWN_CHAR)
        rows.append(line.rstrip("-"))   # trim trailing air

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Convert a Mega Man Maker .mmlv file to VGLC ASCII text.")
    p.add_argument("input", type=Path, help="Input .mmlv file")
    p.add_argument("output", type=Path, nargs="?", help="Output .txt file (default: <input>.txt)")
    args = p.parse_args()

    lines = mmlv_to_vglc(args.input)
    out_path = args.output or args.input.with_suffix(".txt")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote VGLC ASCII ({len(lines)} rows) to {out_path}")


if __name__ == "__main__":
    main()