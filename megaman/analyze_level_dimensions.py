#!/usr/bin/env python3
"""
analyze_level_dimensions.py  (Mega Man / VGLC edition)
======================================================
Measure the width and height of every COMPLETE Mega Man level in a VGLC ASCII
dataset and plot them as a scatter (width = x, height = y) so the distribution
of level sizes across 2D space is visible.

This is the Mega Man counterpart to the Mario Maker version.
The Mario version reads "(source_num)"-delimited levels (many levels per file)
via build_dataset_with_ascii.py. Mega Man VGLC data is laid out differently:
each .txt file is ONE complete level (this is exactly what
bulk_mmlv_to_vglc.py / mmlv_to_vglc.py write into folders like
`megaman/vglc_out`). So instead of splitting files, we read each file as a
single level using the repo's canonical `load_levels` reader -- the same reader
create_megaman_json_data.py uses -- so "a level" here means the same thing it
means downstream.

Width  = number of columns = the longest row in the level.
Height = number of rows = the level's row count after trimming sky on all sides.

Because mmlv_to_vglc.py already crops each level to the tight bounding box of
its real tiles, trimming sky here is normally a no-op on the file dimensions;
it just makes the measurement robust and identical in spirit to the Mario
version.

Sky (empty) characters for Mega Man VGLC -- see datasets/MM.json:
    '-'  passable / empty (air)
    '@'  null (outside the level bounds)
Everything else (#, H, |, P, Z, a, M, ...) is a real placed tile and counts as
content. NOTE: unlike the mm2_tileset_we MegaMan representation, here '-' IS
sky, not a tile.

Usage:
    python megaman/analyze_level_dimensions.py --input megaman/vglc_out --output dist.png
"""

import argparse
import os
import sys

# This script lives in megaman/, but load_levels lives in the repo root module
# create_level_json_data.py, so put the repo root on sys.path before importing.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from pathlib import Path  # noqa: E402
from create_level_json_data import load_levels  # noqa: E402


def level_content_box(rows, empty_chars="-@"):
    """Return one level's CONTENT bounding box as a list of equal-width strings.

    Sky padding is excluded on all four sides: after stripping trailing
    newlines, the empty characters in `empty_chars` are treated as blank, and
    the tightest box that still contains every non-empty tile is cut out. So a
    level whose tiles only span, say, 100 columns and 14 rows comes back as 14
    strings of length 100, not the padded canvas size.

    Ragged rows (shorter than the box) are right-filled with the first empty
    char so every returned row is exactly the box width. Returns [] for a level
    with no content at all.

    In datasets/MM.json the air tile is '-' and '@' is the out-of-bounds null,
    so the default treats both as empty.
    """
    rows = [r.rstrip("\r\n") for r in rows]
    empty = set(empty_chars)
    fill = empty_chars[0] if empty_chars else " "

    def row_has_content(r):
        return any(ch not in empty for ch in r)

    # Vertical extent: first and last rows that hold any non-empty tile.
    top = next((i for i, r in enumerate(rows) if row_has_content(r)), None)
    if top is None:
        return []
    bottom = next(i for i in range(len(rows) - 1, -1, -1) if row_has_content(rows[i]))
    content_rows = rows[top: bottom + 1]

    # Horizontal extent: leftmost and rightmost columns holding a non-empty
    # tile across the content rows.
    left = min(
        next(j for j, ch in enumerate(r) if ch not in empty)
        for r in content_rows if row_has_content(r)
    )
    right = max(
        max(j for j, ch in enumerate(r) if ch not in empty)
        for r in content_rows if row_has_content(r)
    )
    width = right - left + 1
    return [r[left: right + 1].ljust(width, fill) for r in content_rows]


def level_dimensions(rows, empty_chars="-@"):
    """Return (width, height) of one level's content bounding box.

    Thin wrapper over level_content_box. Returns (0, 0) for a level with no
    content at all.
    """
    box = level_content_box(rows, empty_chars=empty_chars)
    if not box:
        return 0, 0
    return len(box[0]), len(box)


def collect_dimensions(input_path, empty_chars="-@"):
    """Walk a folder of VGLC .txt levels (or a single .txt file), returning a
    list of (name, width, height) for each complete level found.

    Each .txt file is one complete level. We read content with the repo's
    canonical `load_levels` so the measurement matches what the dataset builder
    sees, and pair each level with its filename via the identical sorted glob.
    """
    path = Path(input_path)
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        levels = load_levels(str(path))  # same sorted *.txt order as `files`
    else:
        files = [path]
        with open(path, "r", encoding="utf-8") as f:
            levels = [[line.strip() for line in f if line.strip()]]

    results = []
    for file, rows in zip(files, levels):
        w, h = level_dimensions(rows, empty_chars=empty_chars)
        if w == 0 or h == 0:
            continue
        results.append((file.stem, w, h))
    return results


def _summary(values):
    import numpy as np

    arr = np.asarray(values)
    return (
        f"min {arr.min()}, max {arr.max()}, "
        f"mean {arr.mean():.1f}, median {np.median(arr):.0f}"
    )


def make_plot(dims, output_path, title=None):
    import matplotlib

    matplotlib.use("Agg")  # headless: never pop a window, just write the file
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from collections import Counter
    import numpy as np

    widths = [w for _, w, _ in dims]
    heights = [h for _, _, h in dims]

    # Widths/heights are whole tiles, so hundreds of levels can land on the exact
    # same (w, h) and hide behind a single dot. Collapse to one point per unique
    # size and let the count drive both the marker area and the colour, so a
    # popular size reads as a big bright dot instead of vanishing under the pile.
    counts = Counter(zip(widths, heights))
    xs = np.array([w for (w, _h) in counts])
    ys = np.array([h for (_w, h) in counts])
    n = np.array([counts[(w, h)] for (w, h) in counts])

    # Area scales with the square root of the count so a cell with 100 levels
    # isn't 100x the radius of a cell with one -- that would swamp the plot.
    sizes = 25 + 200 * np.sqrt(n / n.max())

    # Extra-wide so the closely-spaced sizes spread out and small differences
    # between sizes are readable.
    fig, ax = plt.subplots(figsize=(19, 9))
    cmap = plt.get_cmap("viridis")
    sc = ax.scatter(
        xs, ys, s=sizes, c=n, cmap=cmap,
        norm=LogNorm(vmin=1, vmax=n.max()),
        alpha=0.85, edgecolors="black", linewidths=0.3,
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Number of levels at this exact size")

    ax.set_xlabel("Level width (columns / tiles)")
    ax.set_ylabel("Level height (rows / tiles)")
    ax.set_title(title or "Complete Mega Man level size distribution")
    ax.grid(True, linestyle=":", alpha=0.4)
    # Give the points a little breathing room from the axes so the busiest
    # cells aren't clipped against the frame.
    ax.margins(0.04)

    # A short stats line in the corner so the plot stands on its own.
    stats = (
        f"levels: {len(dims)}\n"
        f"width:  {_summary(widths)}\n"
        f"height: {_summary(heights)}"
    )
    ax.text(
        0.02, 0.98, stats, transform=ax.transAxes, va="top", ha="left",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Scatter-plot the width/height of every complete Mega Man VGLC level."
    )
    parser.add_argument("--input", required=True,
                        help="Path to a VGLC .txt level file or a folder of .txt files "
                             "(e.g. megaman/vglc_out).")
    parser.add_argument("--output", required=True,
                        help="Output PNG path for the scatter plot.")
    parser.add_argument("--title", default=None,
                        help="Optional plot title.")
    parser.add_argument("--csv", default=None,
                        help="Optional path to also dump per-level dimensions as CSV.")
    parser.add_argument("--empty-chars", default="-@",
                        help="Characters treated as sky/air when trimming the content "
                             "bounding box. Default: '-@' (the MM.json air tile and the "
                             "out-of-bounds null). Every other character is a real tile.")
    args = parser.parse_args()

    dims = collect_dimensions(args.input, empty_chars=args.empty_chars)
    if not dims:
        print(f"ERROR: No non-empty levels found in {args.input}")
        sys.exit(1)

    widths = [w for _, w, _ in dims]
    heights = [h for _, _, h in dims]
    print(f"Measured {len(dims)} complete level(s) from {args.input}")
    print(f"  width:  {_summary(widths)}")
    print(f"  height: {_summary(heights)}")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("name,width,height\n")
            for name, w, h in dims:
                safe = name.replace(",", "_")
                f.write(f"{safe},{w},{h}\n")
        print(f"Per-level dimensions written to {args.csv}")

    make_plot(dims, args.output, title=args.title)
    print(f"Scatter plot written to {args.output}")


if __name__ == "__main__":
    main()
