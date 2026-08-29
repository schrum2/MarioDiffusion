"""
identify_mmlv_ids.py  --  helper for reverse-engineering Mega Man Maker object IDs.

The .mmlv object subtype IDs (the `e` field) are NOT documented anywhere public, so the
mmlv_to_vglc.py decode tables were built by placing known objects in the in-game editor and
reading their IDs back. This tool automates that read-back.

Workflow:
  1. In Mega Man Maker, build a level containing ONE of each object you want to identify
     (and nothing else, or note where each is). Save it -- it lands in
     %LOCALAPPDATA%\\MegaMaker\\Levels\\<name>.mmlv
  2. Run:  python megaman/identify_mmlv_ids.py "<path-to>.mmlv"
     It prints every placed object with its (layer, col, row), its (d, e, i) fields, and
     the char mmlv_to_vglc currently maps it to -- so you can match objects to IDs by
     position and see which are still falling back to '#'/'a'/None.
  3. Add the confirmed IDs to the decode tables in mmlv_to_vglc.py (and the reverse
     emitters in vglc_to_mmlv.py), then re-run this tool to verify.

Pass --summary for just the (d,e) frequency table + which IDs are unmapped (useful on a
big real level or a whole corpus directory).
"""
import argparse
import collections
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmlv_to_vglc import (parse_mmlv, classify, mmlv_to_grid,
                          ENEMY_E_TO_CHAR, GIMMICK_E_TO_CHAR, PICKUP_E_TO_CHAR,
                          BOSS_E_TO_CHAR, WATER_E_IDS)


def is_fallback(cell) -> bool:
    """True if this object hits a generic fallback (its specific ID is unmapped)."""
    d = cell.get("d")
    e = cell.get("e")
    ei = int(e) if e is not None else 0
    if d is None:
        # standalone tile object (water etc.) -- unmapped unless a known water id
        return not (e is not None and int(e) in WATER_E_IDS) and cell.get("i") is None
    dc = int(d)
    if dc == 5:
        return ei != 1 and ei not in ENEMY_E_TO_CHAR
    if dc == 6:
        return ei not in GIMMICK_E_TO_CHAR
    if dc == 7:
        return ei not in PICKUP_E_TO_CHAR
    if dc == 8:
        return ei not in BOSS_E_TO_CHAR
    return False


def dump_level(path: Path):
    cells = parse_mmlv(path)
    xs = [k[1] for k in cells]
    ys = [k[2] for k in cells]
    minx, miny = (min(xs), min(ys)) if cells else (0, 0)

    print(f"== {path.name}: {len(cells)} placed objects ==")
    print(f"{'col':>4} {'row':>4} {'layer':>6} | {'d':>3} {'e':>5} {'i':>3} | char | fallback?")
    print("-" * 64)
    for (layer, tx, ty), cell in sorted(cells.items(), key=lambda kv: (kv[0][2], kv[0][1])):
        d = cell.get("d"); e = cell.get("e"); i = cell.get("i")
        ds = str(int(d)) if d is not None else ""
        es = str(int(e)) if e is not None else ""
        isv = str(int(i)) if i is not None else ""
        ch = classify(cell)
        fb = "  <-- UNMAPPED" if is_fallback(cell) else ""
        lay = layer or "main"
        print(f"{tx-minx:>4} {ty-miny:>4} {lay:>6} | {ds:>3} {es:>5} {isv:>3} | {str(ch):>4} |{fb}")

    print("\n-- converted grid --")
    for row in mmlv_to_grid(path):
        print("".join(row))


def summarize(paths):
    de = collections.Counter()
    fb = collections.Counter()
    for p in paths:
        for _key, cell in parse_mmlv(p).items():
            d = cell.get("d"); e = cell.get("e"); i = cell.get("i")
            if d is not None:
                de[(f"d{int(d)}", f"e{int(e)}" if e is not None else "e?")] += 1
            elif i is not None:
                de[(f"i{int(i)}", "")] += 1
            else:
                de[("none", f"e{int(e)}" if e is not None else "e?")] += 1
            if is_fallback(cell):
                cat = "enemy" if d == 5 else "gimmick" if d == 6 else "pickup" if d == 7 else \
                      "boss" if d == 8 else "standalone"
                fb[(cat, int(e) if e is not None else -1)] += 1
    print(f"scanned {len(paths)} file(s)\n")
    print("== all (d,e) object frequencies ==")
    for k, n in de.most_common(60):
        print(f"  {k}: {n}")
    print("\n== UNMAPPED ids (collapsed to a generic char), most common ==")
    for (cat, e), n in fb.most_common(60):
        print(f"  {cat} e={e}: {n}")


def main():
    ap = argparse.ArgumentParser(description="Inspect Mega Man Maker .mmlv object IDs.")
    ap.add_argument("path", help=".mmlv file, or a directory of them (with --summary)")
    ap.add_argument("--summary", action="store_true",
                    help="print (d,e) frequency + unmapped-id tables instead of a full dump")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_dir():
        paths = [Path(f) for f in glob.glob(str(p / "*.mmlv"))]
        summarize(paths)
    elif args.summary:
        summarize([p])
    else:
        dump_level(p)


if __name__ == "__main__":
    main()
