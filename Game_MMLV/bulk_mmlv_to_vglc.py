from pathlib import Path
import argparse
import re
from tqdm import tqdm
from mmlv_to_vglc import mmlv_to_grid

# Discontiguous levels: some Mega Man Maker levels are separate areas linked only by
# teleporters, with out-of-bounds void ('@') screens between them. Cutting training scenes from
# the whole level treats those areas as one map, so by default each such level is written as its
# contiguous blobs <id>_<k>.txt instead of <id>.txt. mmlv_to_grid pads every level to whole 16x14
# screens, each either entirely void or real content, so a blob is a flood fill over occupied
# screens joined through shared edges (screens touching only diagonally are separate -- the
# player can't move between them). Each blob is cropped to its bounding box in whole screens,
# keeping the screen grid aligned for screen_grid scanning, with any screen in the box that isn't
# part of the blob set to void. Blob 1 holds the spawn ('P'); the rest are numbered top to
# bottom, then left to right, by their first screen.
VOID = "@"
SPAWN = "P"
SCREEN_W, SCREEN_H = 16, 14   # one Mega Man Maker screen, in tiles
BLOB_NAME = re.compile(r"^(.+)_(\d+)$")   # <id>_<k> stem of a blob


def find_blobs(rows):
    """Return the level's blobs as lists of (screen_col, screen_row), spawn blob first, the
    rest in reading order of their first screen."""
    occupied = {
        (sx, sy)
        for sy in range(len(rows) // SCREEN_H)
        for sx in range(len(rows[0]) // SCREEN_W)
        if any(ch != VOID
               for r in rows[sy * SCREEN_H:(sy + 1) * SCREEN_H]
               for ch in r[sx * SCREEN_W:(sx + 1) * SCREEN_W])
    }
    blobs, seen = [], set()
    for start in sorted(occupied, key=lambda s: (s[1], s[0])):   # reading order
        if start in seen:
            continue
        blob, stack = [], [start]
        seen.add(start)
        while stack:
            sx, sy = stack.pop()
            blob.append((sx, sy))
            for n in ((sx + 1, sy), (sx - 1, sy), (sx, sy + 1), (sx, sy - 1)):
                if n in occupied and n not in seen:
                    seen.add(n)
                    stack.append(n)
        blobs.append(blob)

    spawn = next(((x // SCREEN_W, y // SCREEN_H)
                  for y, r in enumerate(rows) for x, ch in enumerate(r) if ch == SPAWN), None)
    blobs.sort(key=lambda b: spawn not in b)   # stable: spawn blob first, rest keep reading order
    return blobs


def crop_blob(rows, blob):
    """Crop rows to the blob's screen bounding box, voiding screens outside the blob."""
    cols = [s[0] for s in blob]
    rws = [s[1] for s in blob]
    x0, x1 = min(cols), max(cols) + 1
    y0, y1 = min(rws), max(rws) + 1
    members = set(blob)
    out = []
    for y in range(y0 * SCREEN_H, y1 * SCREEN_H):
        line = []
        for sx in range(x0, x1):
            seg = rows[y][sx * SCREEN_W:(sx + 1) * SCREEN_W]
            line.append(seg if (sx, y // SCREEN_H) in members else VOID * SCREEN_W)
        out.append("".join(line))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
        help="Folder inside MarioDiffusion to save converted VGLC files"
    )
    parser.add_argument(
        "--show_conversions",
        action="store_true",
        help="Print the original per-file status lines (Converted: ...) instead of the default tqdm progress bar"
    )
    parser.add_argument(
        "--keep_discontiguous",
        action="store_true",
        help="Write discontiguous levels whole as <id>.txt instead of splitting them into their "
             "contiguous blobs <id>_<k>.txt (the default)"
    )
    args = parser.parse_args()

    def status(msg):
        """Emit a routine per-file status line: shown only with --show_conversions, otherwise the
        progress bar conveys progress. Routed through tqdm.write so it never clobbers an active bar."""
        if args.show_conversions:
            tqdm.write(msg)

    # where downloaded levels already are
    input_dir = Path.home() / "AppData/Local/MegaMaker/Levels"

    # user chooses output folder in repo
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.mmlv"))

    print("Found", len(files), "levels")

    # Split levels are also kept whole here, next to --output, for reference. The dataset is
    # built from --output alone, so these never feed it.
    discontiguous_dir = output_dir.parent / f"{output_dir.name}_Discontiguous"
    if not args.keep_discontiguous:
        discontiguous_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    discontiguous_ids = []
    total_blobs = 0

    # A level is written either whole or as blobs, never both, so each level replaces whichever
    # form an earlier run (with or without --keep_discontiguous) left behind. Index the output
    # folder once up front: level stem -> its existing <id>.txt / <id>_<k>.txt files.
    existing = {}
    for p in output_dir.glob("*.txt"):
        m = BLOB_NAME.match(p.stem)
        existing.setdefault(m.group(1) if m else p.stem, []).append(p)

    # By default show a tqdm progress bar over the files; --show_conversions disables it and
    # restores the original per-file "Converted:" prints.
    for file in tqdm(files, desc="Converting levels", unit="level", disable=args.show_conversions):
        try:
            rows = ["".join(row) for row in mmlv_to_grid(file)]
            blobs = [] if args.keep_discontiguous else find_blobs(rows)

            for old in existing.get(file.stem, []):
                old.unlink(missing_ok=True)

            whole_text = "\n".join(rows) + "\n"
            if not args.keep_discontiguous:
                # Drop a whole copy left by an earlier run in case the level is no longer split.
                (discontiguous_dir / f"{file.stem}.txt").unlink(missing_ok=True)
            if len(blobs) > 1:
                for k, blob in enumerate(blobs, start=1):
                    (output_dir / f"{file.stem}_{k}.txt").write_text(
                        "\n".join(crop_blob(rows, blob)) + "\n", encoding="utf-8")
                (discontiguous_dir / f"{file.stem}.txt").write_text(whole_text, encoding="utf-8")
                discontiguous_ids.append(file.stem)
                total_blobs += len(blobs)
                status(f"Converted: {file.name} (discontiguous: {len(blobs)} blobs)")
            else:
                (output_dir / f"{file.stem}.txt").write_text(whole_text, encoding="utf-8")
                status(f"Converted: {file.name}")

            success += 1

        except Exception as e:
            failed += 1
            # Failures are worth surfacing even in bar mode, so route them through tqdm.write
            # (rather than status()) so they show regardless of --show_conversions.
            tqdm.write(f"FAILED: {file.name} - {e}")

    print("\nDone")
    print("Success:", success)
    print("Failed:", failed)
    if not args.keep_discontiguous:
        print(f"Split {len(discontiguous_ids)} discontiguous levels into {total_blobs} blobs "
              f"(--keep_discontiguous writes them whole); whole copies saved to {discontiguous_dir}")


if __name__ == "__main__":
    main()