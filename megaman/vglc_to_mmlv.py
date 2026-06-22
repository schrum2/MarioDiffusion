"""
vglc_to_mmlv.py

Convert a VGLC Mega Man ASCII level (.txt) to a Mega Man Maker level (.mmlv).
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

TILE_PX = 16


def solid_block(x: int, y: int) -> List[str]:
    return [
        f'k{x},{y}="71.000000"',
        f'j{x},{y}="71.000000"',
        f'i{x},{y}="1.000000"',
        f'e{x},{y}="3.000000"',
        f'a{x},{y}="1.000000"',
    ]

def ladder_tile(x: int, y: int) -> List[str]:
    return [
        f'i{x},{y}="3.000000"',
        f'e{x},{y}="98.000000"',
        f'a{x},{y}="1.000000"',
    ]

def spike_tile(x: int, y: int) -> List[str]:
    return [
        f'l{x},{y}="4.000000"',
        f'i{x},{y}="2.000000"',
        f'e{x},{y}="7.000000"',
        f'a{x},{y}="1.000000"',
    ]

def breakable_tile(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="9.000000"',
        f'd{x},{y}="6.000000"',
        f'a{x},{y}="1.000000"',
    ]

def moving_platform(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'h{x},{y}="2.000000"',
        f'e{x},{y}="31.000000"',
        f'd{x},{y}="6.000000"',
        f'a{x},{y}="1.000000"',
    ]

def water_tile(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="177.000000"',
        f'a{x},{y}="1.000000"',
    ]

def orb_tile(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="15.000000"',
        f'd{x},{y}="8.000000"',
        f'a{x},{y}="1.000000"',
    ]

def player_tile(x: int, y: int) -> List[str]:
    spawn_y = max(0, y - TILE_PX)
    return [
        f'o{x},{spawn_y}="9999.000000"',
        f'e{x},{spawn_y}="0.000000"',
        f'd{x},{spawn_y}="4.000000"',
        f'a{x},{spawn_y}="1.000000"',
    ]

def enemy_ground(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="0.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_flying(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="63.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_octopus_lr(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="1.000000"',
        f'd{x},{y}="5.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_octopus_ud(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'g{x},{y}="270.000000"',
        f'e{x},{y}="1.000000"',
        f'd{x},{y}="5.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_beak(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="2.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_picket(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="3.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_screw(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="4.000000"',
        f'd{x},{y}="5.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_big_eye(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="5.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_spine(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="48.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_crazy_razy(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="49.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_watcher(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="52.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_killer_bullet(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="56.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_killer_spawner(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="57.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_tackle_fire(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'h{x},{y}="3.000000"',
        f'e{x},{y}="58.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_flying_shell(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="59.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_flying_shell_spawner(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="60.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_footholder(x: int, y: int) -> List[str]:
    to_x = x + 4 * TILE_PX
    to_y = y + TILE_PX
    return [
        f'o{x},{y}="9999.000000"',
        f'n{x},{y}="{to_y}"',
        f'm{x},{y}="{to_x}"',
        f'e{x},{y}="45.000000"',
        f'd{x},{y}="5.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_jumper(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="159.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

def enemy_gunner(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="7.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]


CHAR_MAP = {
    '@': None,
    '-': None,
    '#': solid_block,
    'A': solid_block,
    't': solid_block,
    '|': ladder_tile,
    'H': spike_tile,
    'B': breakable_tile,
    'M': moving_platform,
    '~': water_tile,
    'Z': orb_tile,
    'P': player_tile,
    'C': solid_block,
    '+': [], 'L': [], 'l': [], 'W': [], 'w': [],
    'D': [], 'U': [], '*': [],
    'a': enemy_ground,
    'b': enemy_flying,
    '<': enemy_octopus_lr,
    '^': enemy_octopus_ud,
    'c': enemy_beak,
    'd': enemy_picket,
    'e': enemy_screw,
    'f': enemy_big_eye,
    'g': enemy_spine,
    'h': enemy_crazy_razy,
    'i': enemy_watcher,
    'j': enemy_killer_bullet,
    'k': enemy_killer_spawner,
    'm': enemy_tackle_fire,
    'n': enemy_flying_shell,
    'o': enemy_flying_shell_spawner,
    'p': enemy_footholder,
    'q': enemy_jumper,
    'r': enemy_gunner,
}

VOID_CHAR = '@'


def compute_playable_col_range(rows: List[str]) -> tuple[int, int]:
    """
    Return (min_col, max_col) of the leftmost and rightmost non-void, non-empty
    column across ALL rows.  Used to decide whether a void cell in the middle of
    the map should be treated as air instead.
    """
    min_col = float('inf')
    max_col = float('-inf')
    for row in rows:
        for col_idx, ch in enumerate(row):
            if ch != VOID_CHAR:
                min_col = min(min_col, col_idx)
                max_col = max(max_col, col_idx)
    if min_col == float('inf'):
        return (0, 0)
    return (int(min_col), int(max_col))


def row_has_non_void(row: str) -> bool:
    return any(ch != VOID_CHAR for ch in row)


def compute_playable_row_range(rows: List[str]) -> tuple[int, int]:
    """
    Return (first_row, last_row) indices of rows that contain at least one
    non-void tile.
    """
    first = None
    last = None
    for idx, row in enumerate(rows):
        if row_has_non_void(row):
            if first is None:
                first = idx
            last = idx
    if first is None:
        return (0, 0)
    return (first, last)

def is_void_enclosed(rows: List[str], row_idx: int, col_idx: int,
                     playable_row_range: tuple[int, int],
                     col_ranges_per_row: List[tuple[int, int] | None]) -> bool:
    """
    A '@' cell is treated as air (enclosed void) if:
      - Its row is within the vertical playable range AND
      - Its column is within the horizontal span of non-void tiles on THAT row
        OR within the span of any immediately neighbouring row.

    This handles the tall black tunnel corridors in level 3 where whole rows
    are '@' but they sit between two playable sections vertically.
    """
    # If the entire row is void, never treat any cell in it as enclosed
    if all(ch == VOID_CHAR for ch in rows[row_idx]):
        return False

    first_row, last_row = playable_row_range
    if row_idx < first_row or row_idx > last_row:
        return False

    # Check this row's own col range
    own = col_ranges_per_row[row_idx]
    if own is not None and own[0] <= col_idx <= own[1]:
        return True

    # Check neighbours (one row above / below) — handles fully-void rows
    for neighbour in (row_idx - 1, row_idx + 1):
        if 0 <= neighbour < len(rows):
            nbr = col_ranges_per_row[neighbour]
            if nbr is not None and nbr[0] <= col_idx <= nbr[1]:
                return True

    return False

def convert(lines: List[str], level_name: str = "Generated", author: str = "converter") -> str:
    rows = [r.rstrip('\n') for r in lines]
    if not rows:
        raise ValueError("Empty level file.")

    # Trim blank @ rows from top only (single row — safe, nothing below shifts)
    while rows and all(ch == '@' for ch in rows[0]):
        rows.pop(0)

    # Trim blank @ rows from bottom in 14-row blocks only
    while len(rows) >= 14 and all(all(ch == '@' for ch in r) for r in rows[-14:]):
        rows = rows[:-14]

    # Pad height to nearest multiple of 14
    current_height = len(rows)
    remainder = current_height % 14
    if remainder != 0:
        width = len(rows[0]) if rows else 0
        rows.extend(['-' * width] * (14 - remainder))


    # Pre-compute per-row column ranges of non-void content
    col_ranges_per_row: List[tuple[int, int] | None] = []
    for row in rows:
        cols = [c for c, ch in enumerate(row) if ch != VOID_CHAR]
        col_ranges_per_row.append((min(cols), max(cols)) if cols else None)

    playable_row_range = compute_playable_row_range(rows)

    out: List[str] = ['[Level]']
    active_screen_rows: set = set()

    for row_idx, row in enumerate(rows):
        y = row_idx * TILE_PX

        for col_idx, ch in enumerate(row):
            x = col_idx * TILE_PX

            # ----------------------------------------------------------------
            # Void handling: skip UNLESS the cell is enclosed within the
            # playable area (the tall-tunnel case in level 3).
            # ----------------------------------------------------------------
            if ch == VOID_CHAR:
                if is_void_enclosed(rows, row_idx, col_idx,
                                    playable_row_range, col_ranges_per_row):
                    # Treat as air — emit 2a only
                    out.append(f'2a{x},{y}="1.000000"')
                    screen_y = (y // 224) * 224
                    active_screen_rows.add(screen_y)
                # else: true outer void — emit nothing
                continue

            emitter = CHAR_MAP.get(ch)
            if emitter is None:
                # Unknown char — treat as air
                out.append(f'2a{x},{y}="1.000000"')
            else:
                if callable(emitter):
                    out.extend(emitter(x, y))
                out.append(f'2a{x},{y}="1.000000"')

            screen_y = (y // 224) * 224
            active_screen_rows.add(screen_y)

    # 2b screen boundary markers
    # 2b screen boundary markers — one per screen-sized block that has content
    screen_rows = set()
    for row_idx, row in enumerate(rows):
        y = row_idx * TILE_PX
        screen_y = (y // 224) * 224
        for col_idx, ch in enumerate(row):
            if ch != VOID_CHAR:
                screen_rows.add(screen_y)

    print(f"DEBUG rows height: {len(rows)}")
    print(f"DEBUG screen_rows: {sorted(screen_rows)}")

    for sy in sorted(screen_rows):
        out.append(f'2b0,{sy}="0.000000"')

    out += [
        '1t="0.000000"',
        '1s="4480.000000"',
        '1r="0.000000"',
        '1q="12800"',
        '1p="0.000000"',
        '1m="9.000000"',
        '1l="11.000000"',
        '1k2="11.000000"',
        '1k1="51.000000"',
        '1k0="0.000000"',
        '1bc="0.000000"',
        '1f="-1.000000"',
        '1e="29.000000"',
        '1d="6.000000"',
        '1bb="0.000000"',
        '1ca="0.000000"',
        '1ba="0.000000"',
        '1c="1.000000"',
        '1b="1.000000"',
        '4b="64.000000"',
        f'4a="{author}"',
        f'1a="{level_name}"',
        '0v="1.6.3"',
        '0a="408382.000000"',
    ]

    return '\n'.join(out) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VGLC Mega Man .txt to .mmlv")
    parser.add_argument("input", type=Path, help="Input VGLC .txt file")
    parser.add_argument("output", type=Path, nargs="?", help="Output .mmlv (default: <input>.mmlv)")
    parser.add_argument("--name", default="GeneratedLevel", help="Level name")
    parser.add_argument("--author", default="converter", help="Author name")
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    lines = text.splitlines()
    out_path = args.output or args.input.with_suffix(".mmlv")
    result = convert(lines, level_name=args.name, author=args.author)
    out_path.write_text(result, encoding="utf-8", newline="\n")
    print(f"Written: {out_path}  ({result.count(chr(10))} lines)")


if __name__ == "__main__":
    main()