"""
mmlv_to_vglc.py
---------------
Convert a Mega Man Maker .mmlv file back to a VGLC-style ASCII text file.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

TILE_PX = 16

# Enemy e-code → VGLC character (only used when d=5 and o=9999)
ENEMY_E_TO_CHAR: Dict[int, str] = {
    0:   'a',   # Met
    63:  'b',   # Fly Boy
    1:   '<',   # Octopus Battery LR  (overridden to '^' if g key present)
    2:   'c',   # Beak
    3:   'd',   # Picket Man
    4:   'e',   # Screw Bomber
    5:   'f',   # Big Eye
    48:  'g',   # Spine
    49:  'h',   # Crazy Razy
    52:  'i',   # Watcher
    56:  'j',   # Killer Bullet
    57:  'k',   # Killer Bullet Spawner
    58:  'm',   # Tackle Fire
    59:  'n',   # Flying Shell
    60:  'o',   # Flying Shell Spawner
    45:  'p',   # Footholder
    159: 'q',   # Jumper
    7:   'r',   # Gunner (d=5 context; note e=7 + i=2 + l=4 is spike, handled separately)
    20:  'b',   # Fly Boy variant (some versions use 20)
    117: 'b',   # another Fly Boy code seen in other levels
}


def parse_mmlv(path: Path) -> Dict[Tuple[int, int], Dict[str, str]]:
    """
    Parse .mmlv and return a dict mapping (x_px, y_px) → {key: value_str}.
    Only coordinate-based entries (those with x,y) are included.
    """
    coord_re = re.compile(r'^([a-zA-Z0-9]+?)(-?\d+),(-?\d+)="([^"]*)"$')
    coords: Dict[Tuple[int, int], Dict[str, str]] = defaultdict(dict)

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        m = coord_re.match(line)
        if not m:
            continue
        key, x, y, val = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        coords[(x, y)][key] = val

    return dict(coords)


def classify_tile(kv: Dict[str, str]) -> str:
    """
    Given the key→value dict at one tile coordinate, return the VGLC character.
    """
    # Helper: get numeric value of a key, or None
    def num(k: str):
        v = kv.get(k)
        return float(v) if v is not None else None

    d = num('d')
    e = num('e')
    i_val = num('i')
    o = num('o')
    l_val = num('l')
    g = num('g')
    k_val = num('k')

    if i_val == 1.0 and e == 3.0 and k_val is not None:
        return '#'

    if i_val == 3.0 and e == 98.0 and l_val is None:
        return 'H'

    if i_val == 2.0 and e == 7.0 and l_val is not None:
        return 'H'

    if o == 9999.0:
        # Player spawn
        if d == 4.0 and e is None:
            return 'P'

        if d == 8.0 and e == 15.0:
            return 'Z'

        if d == 6.0:
            if e == 9.0 or e == 45.0:
                return 'B'
            if e == 31.0:
                return 'M'  

        # Water
        if e == 177.0 and d is None:
            return '~'

        # Enemy (d=5)
        if d == 5.0 and e is not None:
            e_int = int(e)
            if e_int == 1:
                return '^' if g == 270.0 else '<'
            return ENEMY_E_TO_CHAR.get(e_int, '?')

    if set(kv.keys()) <= {'2b'}:
        return '@'

    if '2a' in kv and set(kv.keys()) <= {'2a', '2b'}:
        return '-'

    if '2a' in kv:
        return '-'

    return '@'


def mmlv_to_vglc(path: Path) -> List[str]:
    coord_map = parse_mmlv(path)

    if not coord_map:
        return []

    all_x = [x for x, y in coord_map]
    all_y = [y for x, y in coord_map]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Convert pixel coords to tile indices (may not start at 0)
    # We want col/row starting at 0 for the output
    # But the VGLC format expects the level to start at tile (0,0) with '@' void
    # filling any gap at the top-left.
    # Use the actual grid extent: col = x//16, row = y//16
    max_col = max_x // TILE_PX
    max_row = max_y // TILE_PX

    # Build a lookup by tile col/row
    tile_grid: Dict[Tuple[int, int], str] = {}
    for (x, y), kv in coord_map.items():
        col = x // TILE_PX
        row = y // TILE_PX
        tile_grid[(col, row)] = classify_tile(kv)

    # Render rows
    rows: List[str] = []
    for row in range(max_row + 1):
        line_chars = []
        for col in range(max_col + 1):
            ch = tile_grid.get((col, row), '@')   # absent tile = void
            line_chars.append(ch)
        # Trim trailing void ('@') from each row
        line = ''.join(line_chars).rstrip('@')
        rows.append(line)

    # Trim trailing all-void rows from the bottom
    while rows and set(rows[-1]) <= {'@', ''}:
        rows.pop()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Mega Man Maker .mmlv file to a VGLC ASCII text file."
    )
    parser.add_argument("input", type=Path, help="Input .mmlv file")
    parser.add_argument(
        "output", type=Path, nargs="?",
        help="Output .txt file (default: <input>.txt)"
    )
    args = parser.parse_args()

    lines = mmlv_to_vglc(args.input)
    out_path = args.output or args.input.with_suffix(".txt")
    out_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {len(lines)} rows to {out_path}")


if __name__ == "__main__":
    main()