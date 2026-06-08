"""Convert a VGLC-format ASCII Mega Man level into a simple MMLV JSON format.

This implementation produces a JSON file with the following structure:

{
  "format": "mmlv-json-1",
  "width": <int>,
  "height": <int>,
  "tiles": [[int, ...], ...],
  "mapping": { "char": id, ... }
}

The mapping assigns a small integer id to each distinct character found
in the VGLC input. This is a minimal, clean and Python-native replacement
for the older Java-based MM-NEAT converters; it focuses on the two core
operations requested: VGLC -> MMLV and MMLV -> VGLC (lossy round-trip).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict


def read_vglc(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8").splitlines()
    return [line.rstrip("\n") for line in text]


def vglc_to_mmlv(lines: List[str]) -> Dict:
    if not lines:
        return {"format": "mmlv-json-1", "width": 0, "height": 0, "tiles": [], "mapping": {}}

    height = len(lines)
    width = max(len(l) for l in lines)

    # Normalize lines to same width with space
    norm = [l.ljust(width, " ") for l in lines]

    # Create mapping from char to id
    chars = sorted({ch for line in norm for ch in line})
    mapping = {ch: i for i, ch in enumerate(chars)}

    tiles = [[mapping[ch] for ch in line] for line in norm]

    return {
        "format": "mmlv-json-1",
        "width": width,
        "height": height,
        "tiles": tiles,
        "mapping": mapping,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Convert VGLC ASCII Mega Man level to MMLV JSON.")
    p.add_argument("input", type=Path, help="Input VGLC .txt file")
    p.add_argument("output", type=Path, nargs="?", help="Output .mmlv.json file (optional)")
    args = p.parse_args()

    lines = read_vglc(args.input)
    mmlv = vglc_to_mmlv(lines)

    out_path = args.output or args.input.with_suffix(args.input.suffix + ".mmlv.json")
    out_path.write_text(json.dumps(mmlv, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote MMLV JSON to {out_path}")


if __name__ == "__main__":
    main()
