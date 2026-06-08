"""Convert a simple MMLV JSON file back to a VGLC-style ASCII text file.

This is intentionally lossy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def mmlv_to_vglc(data: Dict) -> List[str]:
    tiles: List[List[int]] = data.get("tiles", [])
    mapping: Dict[str, int] = data.get("mapping", {})

    rev = {int(v): k for k, v in mapping.items()}

    lines: List[str] = []
    for row in tiles:
        line = "".join(rev.get(int(tile), " ") for tile in row)
        lines.append(line.rstrip())

    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="Convert MMLV JSON to VGLC ASCII text.")
    p.add_argument("input", type=Path, help="Input .mmlv.json file")
    p.add_argument("output", type=Path, nargs="?", help="Output .txt file (optional)")
    args = p.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    lines = mmlv_to_vglc(data)

    out_path = args.output or args.input.with_suffix(".reconstructed.txt")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote VGLC ASCII to {out_path}")


if __name__ == "__main__":
    main()
