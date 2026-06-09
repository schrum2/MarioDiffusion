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
    return [
        '1t="0.000000"',
        f'o{x},{y}="9999.000000"',
        f'd{x},{y}="4.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Ground enemy (Met)
def enemy_ground(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="0.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Flying enemy (Fly Boy)
def enemy_flying(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="63.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Octopus Battery left/right
def enemy_octopus_lr(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="1.000000"',
        f'd{x},{y}="5.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Octopus Battery up/down
def enemy_octopus_ud(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'g{x},{y}="270.000000"',
        f'e{x},{y}="1.000000"',
        f'd{x},{y}="5.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Beak (wall enemy)
def enemy_beak(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="2.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Picket Man
def enemy_picket(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="3.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Screw Bomber
def enemy_screw(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="4.000000"',
        f'd{x},{y}="5.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Big Eye
def enemy_big_eye(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="5.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Spine
def enemy_spine(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="48.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Crazy Razy
def enemy_crazy_razy(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="49.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Watcher
def enemy_watcher(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="52.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Killer Bullet
def enemy_killer_bullet(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="56.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Killer Bullet Spawner
def enemy_killer_spawner(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="57.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Tackle Fire
def enemy_tackle_fire(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'h{x},{y}="3.000000"',
        f'e{x},{y}="58.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Flying Shell
def enemy_flying_shell(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="59.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Flying Shell Spawner
def enemy_flying_shell_spawner(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="60.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Footholder (moving platform enemy) - needs destination coords
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

# Jumper
def enemy_jumper(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="159.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]

# Gunner
def enemy_gunner(x: int, y: int) -> List[str]:
    return [
        f'o{x},{y}="9999.000000"',
        f'e{x},{y}="7.000000"',
        f'd{x},{y}="5.000000"',
        f'b{x},{y}="-1.000000"',
        f'a{x},{y}="1.000000"',
    ]


# ---------------------------------------------------------------------------
# Char → entity-line emitter mapping
# Returns None for void/air (caller decides whether to skip or emit 2a only)
# ---------------------------------------------------------------------------

CHAR_MAP = {
    '@': None,          # void – emit nothing at all
    '-': [],            # air – 2a only
    '#': solid_block,
    'A': solid_block,   # appearing block → solid
    't': solid_block,   # pass-through solid → solid
    '|': ladder_tile,   # vertical pipe (some levels use | for ladders)
    'H': ladder_tile,
    'B': breakable_tile,
    'M': moving_platform,
    '~': water_tile,
    'Z': orb_tile,
    'P': player_tile,
    'C': solid_block,   # cannon – treat as solid block visually
    # pickups / items → air (2a only, no entity)
    '+': [], 'L': [], 'l': [], 'W': [], 'w': [],
    'D': [], 'U': [], '*': [],
    # enemies
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


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert(lines: List[str], level_name: str = "Generated", author: str = "converter") -> str:
    rows = [r.rstrip('\n') for r in lines]
    if not rows:
        raise ValueError("Empty level file.")

    out: List[str] = ['[Level]']

    # Track screen-row y values that need 2b markers (for non-void content)
    active_screen_rows: set = set()

    for row_idx, row in enumerate(rows):
        y = row_idx * TILE_PX

        for col_idx, ch in enumerate(row):
            x = col_idx * TILE_PX

            if ch == '@':
                # Void – emit absolutely nothing
                continue

            emitter = CHAR_MAP.get(ch)
            if emitter is None:
                # Unknown char – treat as air (shouldn't normally happen)
                out.append(f'2a{x},{y}="1.000000"')
                continue

            # Emit entity lines (empty list = air tile)
            if callable(emitter):
                out.extend(emitter(x, y))
            # else emitter == [] meaning air, nothing to add

            # Always emit 2a for non-void tiles
            out.append(f'2a{x},{y}="1.000000"')

            # Track which screen rows are active (for 2b markers)
            # Screen rows are multiples of 224px (14 tiles)
            screen_y = (y // 224) * 224
            active_screen_rows.add(screen_y)

    # 2b screen boundary markers – emit in descending order then 0, matching reference
    # Reference has: 2b0,896 (twice), 2b0,672, 2b0,448, 2b0,224, 2b0,0
    sorted_screens = sorted(active_screen_rows, reverse=True)
    # The reference duplicates the first (highest) entry
    if sorted_screens:
        out.append(f'2b0,{sorted_screens[0]}="0.000000"')  # duplicate
    for sy in sorted_screens:
        out.append(f'2b0,{sy}="0.000000"')

    # Hardcoded metadata matching the reference MegaMan 1 level format
    out += [
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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