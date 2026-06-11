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

ENEMY_E_TO_CHAR: Dict[int, str] = {
    0:   'a',
    63:  'b',
    1:   '<',   # overridden to '^' if g=270
    2:   'c',
    3:   'd',
    4:   'e',
    5:   'f',
    48:  'g',
    49:  'h',
    52:  'i',
    56:  'j',
    57:  'k',
    58:  'm',
    59:  'n',
    60:  'o',
    45:  'p',
    159: 'q',
    7:   'r',   # Gunner — e=7 with d=5 (NOT spikes; spikes use i=2)
    20:  'b',
    117: 'b',
}


def parse_mmlv(path: Path) -> Dict[Tuple[int, int], Dict[str, str]]:
    coord_re = re.compile(r'^([a-zA-Z0-9]+?)(-?\d+),(-?\d+)="([^"]*)"$')
    coords: Dict[Tuple[int, int], Dict[str, str]] = defaultdict(dict)

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        m = coord_re.match(line)
        if not m:
            continue
        key, x, y, val = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        coords[(x, y)][key] = val

    # Remove entries that are ONLY 2b — these are screen boundary markers,
    # not real tiles, and would otherwise create phantom rows/columns
    return {
        pos: kv for pos, kv in coords.items()
        if set(kv.keys()) != {'2b'}
    }


def classify_tile(kv: Dict[str, str]) -> str:
    def num(k: str):
        v = kv.get(k)
        return float(v) if v is not None else None

    d   = num('d')
    e   = num('e')
    i_v = num('i')
    o   = num('o')
    l_v = num('l')
    g   = num('g')
    k_v = num('k')

    # Solid block: i=1, e=3, k present
    if i_v == 1.0 and e == 3.0 and k_v is not None:
        return '#'

    # Ladder: i=3, e=98
    if i_v == 3.0 and e == 98.0:
        return '|'

    # Spikes: i=2, e=7 (l key may or may not be present depending on MMM version)
    if i_v == 2.0 and e == 7.0:
        return 'H'

    if o == 9999.0:
        # Player spawn: d=4, no e
        if d == 4.0 and e is None:
            return 'P'

        # Breakable block: d=6, e=9
        if d == 6.0 and e == 9.0:
            return 'B'

        # Moving platform: d=6, e=31
        if d == 6.0 and e == 31.0:
            return 'M'

        # Enemy: d=5
        if d == 5.0 and e is not None:
            e_int = int(e)
            if e_int == 1:
                return '^' if g == 270.0 else '<'
            return ENEMY_E_TO_CHAR.get(e_int, '?')

        # Anything else with o=9999 but no recognised type → air
        return '-'

    # Pure air: only 2a (and maybe 2b)
    if '2a' in kv:
        return '-'

    # Nothing recognised → void
    return '@'


def mmlv_to_vglc(path: Path) -> List[str]:
    coord_map = parse_mmlv(path)

    if not coord_map:
        return []

    all_x = [x for x, y in coord_map]
    all_y = [y for x, y in coord_map]
    max_x = max(all_x)
    max_y = max(all_y)

    max_col = max_x // TILE_PX
    max_row = max_y // TILE_PX

    tile_grid: Dict[Tuple[int, int], str] = {}
    for (x, y), kv in coord_map.items():
        col = x // TILE_PX
        row = y // TILE_PX
        tile_grid[(col, row)] = classify_tile(kv)

    rows: List[str] = []
    for row in range(max_row + 1):
        line_chars = []
        for col in range(max_col + 1):
            ch = tile_grid.get((col, row), '@')
            line_chars.append(ch)
        rows.append(''.join(line_chars))

    # Trim trailing all-void rows from the bottom only
    while rows and set(rows[-1]) <= {'@'}:
        rows.pop()

    # Prepend a leading all-void row to match the original VGLC format,
    # which always has one void row above the first screen of content
    if rows:
        rows.insert(0, '@' * (max_col + 1))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Mega Man Maker .mmlv file to a VGLC ASCII text file."
    )
    parser.add_argument("input", type=Path, help="Input .mmlv file")
    parser.add_argument("output", type=Path, nargs="?",
                        help="Output .txt file (default: <input>.txt)")
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