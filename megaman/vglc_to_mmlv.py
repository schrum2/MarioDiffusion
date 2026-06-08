"""Convert a VGLC-format ASCII Mega Man level into a real Mega Man Maker .mmlv file.

The .mmlv format is a plain-text key=value file understood by Mega Man Maker.
Each tile entry looks like:

    e<x>,<y>="<type_id>.000000"
    a<x>,<y>="1.000000"
    i<x>,<y>="1.000000"
    j<x>,<y>="71.000000"
    k<x>,<y>="71.000000"

Coordinates are in pixels (16 px per tile). The level grid starts at y=64.
The background collision layer (2a / 2c) covers the full rectangular area.

VGLC tile characters -> Mega Man Maker entity IDs:
    -  (empty/air)  -> not written
    X  (solid block) -> 3
    #  (ladder)      -> 9
    S  (spike)       -> 98
    E  (enemy)       -> 2
    P  (player start)-> 0
    Any other char   -> 3  (treated as solid block)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


TILE_SIZE = 16
Y_OFFSET = 64          # level content starts at pixel y=64
SPRITE_ID = 71         # Mega Man 1 tileset sprite sheet ID

# VGLC char -> Mega Man Maker entity ID
CHAR_TO_ID = {
    "-": None,   # air - skip
    "X": 3,      # solid block
    "#": 9,      # ladder
    "S": 98,     # spike
    "E": 2,      # enemy (Met)
    "e": 7,      # enemy variant
    "P": 0,      # player 1 start
    "p": 4,      # player 2 start
}
DEFAULT_SOLID_ID = 3   # unknown chars -> solid block


def read_vglc(path: Path) -> List[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def write_mmlv(lines: List[str], out_path: Path, level_name: str = "Generated Level", author: str = "vglc_to_mmlv") -> None:
    if not lines:
        raise ValueError("VGLC level is empty.")

    height = len(lines)
    width = max(len(l) for l in lines)

    # Normalize all rows to the same width
    norm = [l.ljust(width, "-") for l in lines]

    # Pixel dimensions
    px_width = width * TILE_SIZE
    px_height = height * TILE_SIZE

    entries: List[str] = []

    # --- background collision layer (2a / 2c) ---
    # Covers the full bounding box starting from y=0
    for row in range(height + Y_OFFSET // TILE_SIZE):
        py = row * TILE_SIZE
        for col in range(width):
            px = col * TILE_SIZE
            entries.append(f'2a{px},{py}="1.000000"')
            entries.append(f'2c{px},{py}="1.000000"')

    # --- tile entities ---
    for row_idx, row in enumerate(norm):
        py = Y_OFFSET + row_idx * TILE_SIZE
        for col_idx, ch in enumerate(row):
            entity_id = CHAR_TO_ID.get(ch, DEFAULT_SOLID_ID)
            if entity_id is None:
                continue  # air - no entry needed
            px = col_idx * TILE_SIZE
            entries.append(f'k{px},{py}="{SPRITE_ID}.000000"')
            entries.append(f'j{px},{py}="{SPRITE_ID}.000000"')
            entries.append(f'i{px},{py}="1.000000"')
            entries.append(f'e{px},{py}="{entity_id}.000000"')
            entries.append(f'a{px},{py}="1.000000"')

    # --- level metadata ---
    total_px_size = px_width * (px_height + Y_OFFSET)
    entries.append(f'1s="{float(total_px_size):.6f}"')
    entries.append('1r="0.000000"')
    entries.append(f'1q="{px_width}"')
    entries.append('1p="0.000000"')
    entries.append(f'1m="{height}.000000"')   # height in tiles
    entries.append(f'1l="{width}.000000"')    # width in tiles
    entries.append('1k2="11.000000"')
    entries.append('1k0="0.000000"')
    entries.append('1bc="0.000000"')
    entries.append('1f="-1.000000"')
    entries.append('1e="29.000000"')          # background ID (Wily Castle)
    entries.append('1d="6.000000"')           # music ID
    entries.append('1bb="0.000000"')
    entries.append('1ca="0.000000"')
    entries.append('1ba="0.000000"')
    entries.append('1c="1.000000"')
    entries.append('1b="1.000000"')
    entries.append(f'1t="0.000000"')
    entries.append(f'4b="64.000000"')
    entries.append(f'4a="{author}"')
    entries.append(f'1a="{level_name}"')
    entries.append('0v="1.6.3"')
    entries.append('0a="408500.000000"')

    content = "[Level]\r\n" + "\r\n".join(entries) + "\r\n"
    out_path.write_text(content, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert a VGLC ASCII Mega Man level to a Mega Man Maker .mmlv file.")
    p.add_argument("input", type=Path, help="Input VGLC .txt file")
    p.add_argument("output", type=Path, nargs="?", help="Output .mmlv file (default: <input>.mmlv)")
    p.add_argument("--name", default="Generated Level", help="Level name shown in Mega Man Maker")
    p.add_argument("--author", default="vglc_to_mmlv", help="Author name shown in Mega Man Maker")
    args = p.parse_args()

    lines = read_vglc(args.input)
    out_path = args.output or args.input.with_suffix(".mmlv")
    write_mmlv(lines, out_path, level_name=args.name, author=args.author)
    print(f"Wrote Mega Man Maker level to {out_path}")
    print(f"To play: copy {out_path.name} to:")
    print(r"  C:\Users\<YourName>\AppData\Local\MegaMaker\Levels" + "\\")


if __name__ == "__main__":
    main()