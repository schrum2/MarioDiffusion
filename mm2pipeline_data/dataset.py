"""Build a tile-id dataset from ASCII level files, and split a dataset into
train/validate/test.

Two subcommands when run as a module:
    python -m mm2pipeline_data.dataset build  --input ... --output_folder ... --tileset ...
    python -m mm2pipeline_data.dataset split  --input ...

``build`` parses each .txt file into per-level row lists (split on "(source_num)"
headers), crops each level to a WINDOW_H x WINDOW_W window (best window, or every
window with --sliding_window) and emits a grid of tile ids. Per-level metadata
(the metadata.json sidecar written by ``mm2pipeline_data.ascii to-ascii``) is folded
into each sample, the goal can be stripped with --strip_goal, mostly-air windows
are diverted to a companion "dropped" dataset, and --with_images crops the
matching slice of each level's rendered PNG. ``split`` shuffles a dataset and
writes -train / -validate / -test files next to it.
"""

import argparse
import json
import math
import os
import sys
import re
import random
from pathlib import Path

from . import paths

WINDOW_H = 20
WINDOW_W = 20
EXTRA_TILE = "_"


# ===========================================================================
# build
# ===========================================================================

def load_tileset(path):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: Tileset file not found at {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chars = sorted(data["tiles"].keys())
    # Determine the extra/padding tile for unknown source characters.
    # If "_" is already a real tile in this tileset (e.g., extended_tiles.json
    # defines it as semisolid platform), use a null-byte sentinel instead so
    # unknown chars don't silently become semisolid tiles.
    if EXTRA_TILE not in chars:
        extra_tile = EXTRA_TILE
    else:
        extra_tile = "\x00"
    chars.append(extra_tile)
    return {ch: idx for idx, ch in enumerate(chars)}, extra_tile


def _pad_rows(rows, width, empty_char):
    # Pad short levels at the top (and pad every row out to `width`) so the
    # level stays bottom-aligned in the window -- the ground belongs on the
    # bottom rows, not floating in the middle.
    pad_rows = max(0, WINDOW_H - len(rows))
    padded = [empty_char * width] * pad_rows + list(rows)
    return [r.ljust(width, empty_char) for r in padded]


def extract_best_window(rows, tile_to_id, extra_tile=EXTRA_TILE, empty_char="-"):
    """Return (best_scene, best_x): the single WINDOW_H x WINDOW_W window with
    the most non-empty tiles (the busiest slice of the level) and the column
    offset (in tiles, from the left edge) where that window starts. Returns
    (None, None) when the level is too small to hold a window.

    best_x is the game-grid column of the window's left edge, which is what the
    image cropper needs to line the picture up with the tile sample."""
    extra_id = tile_to_id.get(extra_tile, 0)
    empty_id = tile_to_id.get(empty_char, 0)

    height = len(rows)
    width = max((len(r) for r in rows), default=0)

    if width < WINDOW_W or height == 0:
        return None, None

    padded = _pad_rows(rows, width, empty_char)

    best_score = -1
    best_scene = None
    best_x = None

    for x in range(width - WINDOW_W + 1):
        scene = []
        score = 0
        for y in range(WINDOW_H):
            row_slice = padded[y][x : x + WINDOW_W]
            id_row = []
            for ch in row_slice:
                tid = tile_to_id.get(ch, extra_id)
                id_row.append(tid)
                if tid not in (empty_id, extra_id):
                    score += 1
            scene.append(id_row)

        if score > best_score:
            best_score = score
            best_scene = scene
            best_x = x

    return best_scene, best_x


def extract_all_windows(rows, tile_to_id, extra_tile=EXTRA_TILE, stride=1, empty_char="-"):
    """Slide a WINDOW_H x WINDOW_W window across the level and return every window
    as a list of (x, scene) pairs, where x is the game-grid column of the
    window's left edge (needed to crop the matching slice of the level image).
    Air-only windows are dropped so empty gaps don't end up in the dataset."""
    extra_id = tile_to_id.get(extra_tile, 0)
    empty_id = tile_to_id.get(empty_char, 0)

    width = max((len(r) for r in rows), default=0)

    if width < WINDOW_W or not rows:
        return []

    padded = _pad_rows(rows, width, empty_char)

    # Cover the right edge when stride doesn't divide evenly, but only add a
    # fresh edge window when the tail is big enough to carry new content;
    # otherwise snap the last window to the edge so we don't stack a duplicate.
    last_x = width - WINDOW_W
    xs = list(range(0, last_x + 1, stride))
    if xs[-1] != last_x:
        min_tail = max(1, min(stride, WINDOW_W) // 2)
        if last_x - xs[-1] >= min_tail:
            xs.append(last_x)
        else:
            xs[-1] = last_x

    scenes = []
    for x in xs:
        scene = []
        has_content = False
        for y in range(WINDOW_H):
            row_slice = padded[y][x : x + WINDOW_W]
            id_row = []
            for ch in row_slice:
                tid = tile_to_id.get(ch, extra_id)
                id_row.append(tid)
                if tid != empty_id:
                    has_content = True
            scene.append(id_row)
        if has_content:
            scenes.append((x, scene))

    return scenes


def count_non_air_tiles(scene, empty_id, extra_id):
    """Number of "real" tiles in a window -- everything that isn't the empty
    (sky) tile or the unknown/padding tile. Used to throw out samples that are
    mostly air and carry too little structure to be worth training on."""
    return sum(
        1
        for row in scene
        for tid in row
        if tid not in (empty_id, extra_id)
    )


def parse_source_file(file_path):
    """
    Split a file into per-level row lists keyed by their "(source_num)" header,
    discarding the {{{ / }}} separator lines. Files with no headers are returned
    as a single "source_0" level.
    """
    text = Path(file_path).read_text(encoding="utf-8")
    lines = text.splitlines()

    levels = {}
    current_source = None
    current_rows = []

    for line in lines:
        cleaned_line = line.strip()

        # Skip pure structural markers or blank separator lines
        if not cleaned_line or cleaned_line.startswith("{{{") or cleaned_line.startswith("}}}"):
            continue

        # Match a "(source_num)" header, capturing the id and anything that
        # follows it on the same line.
        match = re.match(r'^\s*\(([^)]*)\)(.*)', line)

        if match:
            # Flush the previous level before starting a new one.
            if current_source and current_rows:
                levels[current_source] = current_rows

            source_num = match.group(1)
            current_source = f"source_{source_num}"

            # Keep any map data that trails the header on the same line.
            map_part = match.group(2)
            if map_part.strip():
                current_rows = [map_part]
            else:
                current_rows = []
        else:
            # Another row of the current level.
            if current_source is not None:
                current_rows.append(line)

    # Flush the last level at EOF.
    if current_source and current_rows:
        levels[current_source] = current_rows

    # No "(source_num)" headers anywhere: treat the whole file as one level.
    if not levels and lines:
        levels["source_0"] = lines

    return levels


def check_unmapped_chars(rows, tile_to_id, extra_tile):
    """
    Count how many characters in rows aren't real keys in the tileset.
    A high ratio almost always means the wrong --tileset was passed for this input
    """
    total = 0
    unmapped = 0
    unmapped_chars = set()
    for row in rows:
        for ch in row:
            total += 1
            if ch not in tile_to_id:
                unmapped += 1
                unmapped_chars.add(ch)
    return total, unmapped, unmapped_chars


def collect_input_files(input_path):
    p = Path(input_path)
    if p.is_dir():
        files = sorted(p.glob("*.txt"))
        if not files:
            sys.exit(f"ERROR: No .txt files found in folder {input_path}")
        return files
    if p.is_file():
        return [p]
    sys.exit(f"ERROR: Input path not found: {input_path}")


def load_level_metadata(metadata_arg, input_path):
    """Load the metadata map (ascii stem -> fields) from --metadata, or a
    metadata.json next to the input. Returns {} if there's none."""
    if metadata_arg:
        path = Path(metadata_arg)
        if not path.is_file():
            sys.exit(f"ERROR: Metadata file not found at {metadata_arg}")
    else:
        p = Path(input_path)
        folder = p if p.is_dir() else p.parent
        path = folder / "metadata.json"
        if not path.is_file():
            return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        sys.exit(f"ERROR: Could not read metadata file {path}: {e}")
    print(f"Loaded metadata for {len(data)} level(s) from {path}")
    return data


def detect_empty_char(tileset_path):
    # Prefer a literal space if the tileset defines one, otherwise fall back
    # to '-' (the VGLC empty-tile glyph).
    with open(tileset_path, encoding="utf-8") as f:
        tiles = json.load(f)["tiles"]
    return " " if " " in tiles else "-"


def detect_goal_chars(tileset_path):
    # Characters the tileset tags as the level goal/flagpole. Falls back to the
    # native MM2 glyph 'G' when the tileset carries no goal tag (e.g. the
    # smb/extended tilesets), so --strip_goal still works on raw MM2 input.
    with open(tileset_path, encoding="utf-8") as f:
        tiles = json.load(f)["tiles"]
    goal = {ch for ch, tags in tiles.items() if "goal" in tags or "flagpole" in tags}
    return goal or {"G"}


def load_converter(filename, module_name):
    """Load one of the repo-root ASCII converter scripts (mm2view_to_extended.py,
    ascii_to_vglc.py) by path."""
    import importlib.util
    path = str(paths.repo_path(filename))
    if not os.path.isfile(path):
        sys.exit(f"ERROR: Converter module missing from {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Level-image sampling (--with_images)
#
# The export renders each level to a PNG in a sibling "images/" folder with the
# same stem as the ascii .txt. Toost renders at 16 px/tile, bottom-left aligned,
# so a tile window cuts straight out of the PNG.
# ---------------------------------------------------------------------------

def _load_pil():
    try:
        from PIL import Image
    except ImportError:
        sys.exit(
            "ERROR: --with_images requires Pillow. Install it with:\n"
            "    pip install Pillow"
        )
    return Image


class ImageLocator:
    """Find the rendered PNG for a level, given the ascii input file and the
    level's stem. Checks the conventional 'images/' folders first, then falls
    back to a one-time recursive scan of the machine (cached) when allowed."""

    def __init__(self, explicit_dir=None, deep_search=True, search_root=None):
        self.explicit_dir = Path(explicit_dir) if explicit_dir else None
        self.deep_search = deep_search
        self.search_root = Path(search_root) if search_root else None
        self._index = None  # lowercase filename -> Path, built lazily on first miss

    def _candidate_dirs(self, input_file):
        p = Path(input_file).resolve()
        dirs = []
        if self.explicit_dir:
            dirs.append(self.explicit_dir)
        # The export layout puts images one folder out from ascii/, but also
        # try a few nearby spots so loose folder structures still resolve.
        dirs.append(p.parent.parent / "images")  # .../<export>/images  (canonical)
        dirs.append(p.parent / "images")
        dirs.append(p.parent.parent.parent / "images")
        dirs.append(p.parent)                     # alongside the ascii file
        return dirs

    def _build_index(self, input_file):
        if self._index is not None:
            return self._index
        root = self.search_root
        if root is None:
            root = Path(Path(input_file).resolve().anchor or Path.home())
        print(f"  [image-search] No image in the usual spots; indexing PNGs under "
              f"{root} (one-time, may be slow)...")
        index = {}
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".png"):
                    # First match wins; don't clobber an earlier (shallower) hit.
                    index.setdefault(fn.lower(), Path(dirpath) / fn)
        self._index = index
        print(f"  [image-search] Indexed {len(index)} PNG file(s).")
        return index

    def find(self, input_file, stem):
        fname = f"{stem}.png"
        for d in self._candidate_dirs(input_file):
            cand = d / fname
            if cand.is_file():
                return cand
        if self.deep_search:
            return self._build_index(input_file).get(fname.lower())
        return None


def crop_image_window(Image, img, x_tile, ppt, fill=(0, 0, 0)):
    """Cut a WINDOW_W x WINDOW_H tile region (at `ppt` pixels/tile) out of the
    level image, bottom-aligned to match the tile sample, starting at column
    `x_tile`. Regions past the image edge are padded with `fill` so the crop is
    always exactly the window size."""
    img = img.convert("RGB")
    img_w, img_h = img.size
    box_w, box_h = WINDOW_W * ppt, WINDOW_H * ppt

    left = x_tile * ppt
    top = img_h - box_h  # bottom-aligned: the ground sits on the bottom rows

    canvas = Image.new("RGB", (box_w, box_h), fill)
    sl, st = max(0, left), max(0, top)
    sr, sb = min(img_w, left + box_w), min(img_h, top + box_h)
    if sr > sl and sb > st:
        region = img.crop((sl, st, sr, sb))
        canvas.paste(region, (sl - left, st - top))
    return canvas


def _safe_filename(name):
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)


def build_sample_entry(sample_name, x, scene, with_images, Image, level_img, ppt,
                       image_out_dir, file_stem, meta=None):
    """Make a dataset entry for one window, cropping its image slice when
    --with_images is on. Returns (entry, saved_image); `meta` goes between the
    name and the scene."""
    entry = {"name": sample_name}
    if meta:
        entry.update(meta)
    entry["scene"] = scene
    if not with_images:
        return entry, False
    if level_img is not None and x is not None:
        crop = crop_image_window(Image, level_img, x, ppt)
        # Prefix the crop with the level name so images never collide.
        if sample_name.startswith(f"{file_stem}_") or sample_name.startswith(f"{file_stem}/"):
            img_stem = sample_name
        else:
            img_stem = f"{file_stem}_{sample_name}"
        crop_path = image_out_dir / f"{_safe_filename(img_stem)}.png"
        crop.save(crop_path)
        # Store the exact (absolute) path so the dataset opens from any cwd.
        entry["image"] = str(crop_path.resolve())
        return entry, True
    entry["image"] = None
    return entry, False


def main_build(argv=None):
    # The window dimensions are read as module globals throughout; declare them
    # global up front so --window_h/--window_w can push the requested sizes back.
    global WINDOW_H, WINDOW_W

    parser = argparse.ArgumentParser(description="Build dataset from custom tagged text files.")
    parser.add_argument("--input", required=True, help="Path to a .txt file or a folder of .txt files.")
    parser.add_argument("--output_folder", required=True, help="Output JSON filename.")
    parser.add_argument("--tileset", required=True, help="Path to tileset JSON.")
    convert_group = parser.add_mutually_exclusive_group()
    convert_group.add_argument("--convert_to_vglc", action="store_true",
                               help="Convert layout to VGLC structure (ascii_to_vglc.py).")
    convert_group.add_argument("--convert_to_extended", action="store_true",
                               help="Convert layout to extended tile format (mm2view_to_extended.py).")
    parser.add_argument("--window_h", type=int, default=WINDOW_H,
                        help=f"Window height in tiles. Default: {WINDOW_H}.")
    parser.add_argument("--window_w", type=int, default=WINDOW_W,
                        help=f"Window width in tiles. Default: {WINDOW_W}.")
    parser.add_argument("--sliding_window", action="store_true",
                        help="Collect every window position as a separate sample instead of keeping only the best window.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Step size (in tiles) between windows when --sliding_window is active. Default: the window width (no overlap).")
    parser.add_argument("--strip_goal", action="store_true",
                        help="Replace the goal/flagpole tile with air (the empty "
                             "tile) before windowing, so levels are encoded without "
                             "their end goal.")
    parser.add_argument("--min_tiles_pct", type=float, default=7.0,
                        help="Drop samples where non-air tiles (sky and unknown "
                             "tiles don't count) make up less than this percent of "
                             "the window. Default: 7 (i.e. 7%%).")
    parser.add_argument("--dropped_output", default=None,
                        help="Where to write a second dataset holding the samples "
                             "rejected by --min_tiles_pct. Default: '<output>_dropped.json'. "
                             "Pass --no_dropped to skip writing it.")
    parser.add_argument("--no_dropped", action="store_true",
                        help="Don't write the 'dropped' dataset of below-min_tiles_pct samples.")
    parser.add_argument("--with_images", action="store_true",
                        help="For every tile sample, also crop the matching "
                             f"{WINDOW_W}x{WINDOW_H}-tile region out of the level's "
                             "rendered PNG and save it next to the dataset. The PNG "
                             "is the one produced by the .bcd export (same stem as "
                             "the ascii file, in a sibling 'images/' folder). This is necessary for use with VLMs.")
    parser.add_argument("--image_dir", default=None,
                        help="Explicit folder to look in for level PNGs (checked "
                             "before the conventional 'images/' locations).")
    parser.add_argument("--image_output_dir", default=None,
                        help="Where to write the cropped sample PNGs when "
                             "--with_images is set. Default: '<output>_images/'.")
    parser.add_argument("--no_image_search", action="store_true",
                        help="Disable the slow machine-wide PNG search fallback; "
                             "only look in the conventional 'images/' folders.")
    parser.add_argument("--image_search_root", default=None,
                        help="Root directory for the machine-wide PNG search "
                             "fallback. Default: the input drive.")
    parser.add_argument("--metadata", default=None,
                        help="Path to the per-level metadata JSON produced by "
                             "'mm2pipeline_data.ascii to-ascii' (keyed by ascii file "
                             "stem, with level_name/difficulty/gamestyle/theme/"
                             "tags). Each sample from a level is tagged with its "
                             "metadata. If omitted, a 'metadata.json' sitting in "
                             "the input folder (or next to a single input file) is "
                             "picked up automatically.")
    args = parser.parse_args(argv)

    # Push the requested sizes into the module globals before any extraction.
    WINDOW_H = args.window_h
    WINDOW_W = args.window_w
    # Default stride is the window width (no overlap), tracking --window_w.
    stride = args.stride if args.stride is not None else WINDOW_W
    # Convert the percentage threshold into a tile count against the actual window size.
    min_tiles = math.ceil((args.min_tiles_pct / 100.0) * WINDOW_H * WINDOW_W)

    if args.with_images and (args.convert_to_vglc or args.convert_to_extended):
        parser.error("--with_images cannot be combined with --convert_to_vglc / "
                     "--convert_to_extended: the rendered PNGs match the native "
                     "MM2 grid, not the converted tile layout.")

    # --convert_to_extended needs the extended glyphs; if the caller left the
    # base smb tileset in place, quietly point it at extended_tiles.json.
    tileset_path = args.tileset
    if args.convert_to_extended and Path(tileset_path).resolve() == paths.SMB_TILESET_PATH.resolve():
        tileset_path = str(paths.EXTENDED_TILESET_PATH)
    tile_to_id, extra_tile = load_tileset(tileset_path)

    default_empty_char = detect_empty_char(tileset_path)
    goal_chars = detect_goal_chars(tileset_path) if args.strip_goal else set()

    converter_mod = None
    if args.convert_to_vglc:
        converter_mod = load_converter("ascii_to_vglc.py", "ascii_to_vglc")
    elif args.convert_to_extended:
        converter_mod = load_converter("mm2view_to_extended.py", "mm2view_to_extended")
        # Reduce onto the same tileset we encode against, so the converter's surviving
        # glyphs are exactly this tileset's ids (e.g. extended_tiles_30.json), not the
        # converter's default extended_tiles.json.
        converter_mod.set_target(tileset_path)

    keep_dropped = not args.no_dropped

    # --with_images setup: locate the rendered PNGs and a place to write crops.
    Image = None
    locator = None
    image_out_dir = None
    dropped_image_out_dir = None
    images_saved = 0
    dropped_images_saved = 0
    images_missing = 0
    if args.with_images:
        Image = _load_pil()
        locator = ImageLocator(
            explicit_dir=args.image_dir,
            deep_search=not args.no_image_search,
            search_root=args.image_search_root,
        )
        if args.image_output_dir:
            image_out_dir = Path(args.image_output_dir)
        else:
            out = Path(args.output_folder)
            image_out_dir = out.parent / f"{out.stem}_images"
        image_out_dir.mkdir(parents=True, exist_ok=True)
        # Crops for the dropped dataset go in a parallel sibling folder so they
        # never get mixed in with the real samples.
        if keep_dropped:
            dropped_image_out_dir = image_out_dir.parent / f"{image_out_dir.name}_dropped"
            dropped_image_out_dir.mkdir(parents=True, exist_ok=True)

    input_files = collect_input_files(args.input)
    level_metadata = load_level_metadata(args.metadata, args.input)
    dataset = []
    dataset_dropped = []     # samples rejected by --min_tiles, kept for inspection
    processed = 0
    skipped = 0
    dropped_min_samples = 0  # samples set aside for being under --min_tiles

    for input_file in input_files:
        raw_levels = parse_source_file(input_file)
        file_stem = input_file.stem
        file_meta = level_metadata.get(file_stem)
        print(f"Parsing content from {input_file}...")

        for name, rows in raw_levels.items():
            # Prefix with the source filename so names stay unique across files
            full_name = f"{file_stem}/{name}" if len(input_files) > 1 else name
            try:
                level_img = None    # the level's rendered PNG, if we found one
                ppt = None          # pixels per tile in that PNG
                if converter_mod is not None:
                    rows = converter_mod.convert_level(rows)
                    empty_char = " "
                else:
                    rows = [r.rstrip('\r\n') for r in rows]
                    # The original (pre-crop) row count maps 1:1 to the image's
                    # tile rows, so it sets the pixels-per-tile scale.
                    orig_row_count = len(rows)
                    while rows and not rows[0].strip():
                        rows.pop(0)
                    if len(rows) > WINDOW_H:
                        rows = rows[-WINDOW_H:]
                    empty_char = default_empty_char

                    if args.with_images:
                        # PNG shares the ascii stem; fall back to the source name.
                        img_path = locator.find(input_file, file_stem)
                        if not img_path and name != file_stem:
                            img_path = locator.find(input_file, name)
                        if img_path and orig_row_count > 0:
                            level_img = Image.open(img_path)
                            ppt = max(1, round(level_img.size[1] / orig_row_count))
                        else:
                            images_missing += 1
                            print(f"  [no-image] {full_name}: no level image found "
                                  f"on this machine; sample(s) kept without an image crop.")

                # Wipe the goal to air (after conversion, so glyphs match `rows`).
                if goal_chars:
                    table = str.maketrans({g: empty_char for g in goal_chars})
                    rows = [r.translate(table) for r in rows]

                total_chars, unmapped_chars_count, unmapped_chars = check_unmapped_chars(rows, tile_to_id, extra_tile)
                if total_chars and unmapped_chars_count / total_chars > 0.2:
                    print(
                        f"  [WARNING] {full_name}: {unmapped_chars_count}/{total_chars} chars "
                        f"({unmapped_chars_count / total_chars:.0%}) not found in "
                        f"'{os.path.basename(tileset_path)}' and will collapse to the unknown tile. "
                        f"Unmapped chars: {' '.join(repr(c) for c in sorted(unmapped_chars))}. "
                        f"You are probably using the wrong --tileset for this input."
                    )

                # Ids that count as "air" for the min-tile filter: sky (empty)
                # and the unknown/padding tile.
                empty_id = tile_to_id.get(empty_char, 0)
                extra_id = tile_to_id.get(extra_tile, 0)

                # (sample_name, x, scene); x is the left-edge column for cropping.
                samples = []
                dropped_samples = []
                if args.sliding_window:
                    windows = extract_all_windows(rows, tile_to_id, extra_tile=extra_tile, stride=stride, empty_char=empty_char)
                    if not windows:
                        print(f"  [SKIP] {full_name} (empty)")
                        skipped += 1
                        if level_img is not None:
                            level_img.close()
                        continue
                    # Split the windows: enough non-air tiles -> keep, otherwise
                    # set aside for the dropped dataset.
                    kept = []
                    dropped_windows = []
                    for x, scene in windows:
                        if count_non_air_tiles(scene, empty_id, extra_id) >= min_tiles:
                            kept.append((x, scene))
                        else:
                            dropped_windows.append((x, scene))
                    dropped_min_samples += len(dropped_windows)
                    dropped_samples = [(f"{full_name}_{i}", x, scene) for i, (x, scene) in enumerate(dropped_windows)]
                    samples = [(f"{full_name}_{i}", x, scene) for i, (x, scene) in enumerate(kept)]
                    suffix = f", {len(dropped_windows)} under {min_tiles} tiles dropped" if dropped_windows else ""
                    print(f"  [OK] {full_name} ({len(samples)} windows{suffix})")
                else:
                    scene, best_x = extract_best_window(rows, tile_to_id, extra_tile=extra_tile, empty_char=empty_char)
                    if scene is None:
                        print(f"  [SKIP] {full_name} (empty)")
                        skipped += 1
                        if level_img is not None:
                            level_img.close()
                        continue
                    if count_non_air_tiles(scene, empty_id, extra_id) < min_tiles:
                        print(f"  [OK] {full_name} (under {min_tiles} tiles, dropped)")
                        dropped_min_samples += 1
                        dropped_samples = [(full_name, best_x, scene)]
                    else:
                        samples = [(full_name, best_x, scene)]
                        print(f"  [OK] {full_name}")

                for sample_name, x, scene in samples:
                    entry, saved = build_sample_entry(
                        sample_name, x, scene, args.with_images, Image,
                        level_img, ppt, image_out_dir, file_stem, meta=file_meta)
                    if saved:
                        images_saved += 1
                    dataset.append(entry)

                if keep_dropped:
                    for sample_name, x, scene in dropped_samples:
                        entry, saved = build_sample_entry(
                            sample_name, x, scene, args.with_images, Image,
                            level_img, ppt, dropped_image_out_dir, file_stem, meta=file_meta)
                        if saved:
                            dropped_images_saved += 1
                        dataset_dropped.append(entry)

                processed += len(samples)
                if level_img is not None:
                    level_img.close()

            except Exception as e:
                print(f"  [ERROR] Failed processing {full_name}: {e}")
                skipped += 1

    # Save output dataset
    output_file = Path(args.output_folder)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    # Save the companion "dropped" dataset of below-min_tiles_pct samples.
    dropped_file = None
    if keep_dropped:
        if args.dropped_output:
            dropped_file = Path(args.dropped_output)
        else:
            dropped_file = output_file.parent / f"{output_file.stem}_dropped{output_file.suffix}"
        dropped_file.parent.mkdir(parents=True, exist_ok=True)
        with open(dropped_file, "w", encoding="utf-8") as f:
            json.dump(dataset_dropped, f, indent=2)

    print(f"\nCompleted! Packaged {processed} items into {output_file} ({skipped} skipped).")
    if dropped_min_samples:
        print(f"Dropped for --min_tiles_pct ({args.min_tiles_pct:g}%, "
              f"{min_tiles} tiles): {dropped_min_samples} sample(s).")
    if keep_dropped:
        print(f"Dropped dataset: {len(dataset_dropped)} sample(s) written to {dropped_file}.")
    if args.with_images:
        print(f"Image crops: {images_saved} saved to {image_out_dir} "
              f"({images_missing} level(s) had no image on this machine).")
        if keep_dropped:
            print(f"Dropped image crops: {dropped_images_saved} saved to {dropped_image_out_dir}.")

# num_tiles = 138 for mm2_tileset_full
# num_tiles = 68 for mm2_tileset_we


# ===========================================================================
# split
# ===========================================================================

def main_split(argv=None):
    parser = argparse.ArgumentParser(description="Split a Mario Maker dataset into train/val/test sets.")
    parser.add_argument("--input", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--train_pct", type=float, default=0.8)
    parser.add_argument("--val_pct", type=float, default=0.1)
    parser.add_argument("--test_pct", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if abs(args.train_pct + args.val_pct + args.test_pct - 1.0) > 1e-6:
        raise ValueError("Train/val/test percentages must sum to 1.0")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)

    n = len(data)
    # Round (not truncate) so val/test aren't zeroed on small datasets, and give
    # each non-zero split at least one sample, borrowing from the largest.
    n_val = max(1, round(args.val_pct * n)) if args.val_pct > 0 and n > 0 else 0
    n_test = max(1, round(args.test_pct * n)) if args.test_pct > 0 and n > 0 else 0
    n_train = n - n_val - n_test
    while n_train < 1 and (n_val > 0 or n_test > 0) and n > 0:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        n_train = n - n_val - n_test

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    base, ext = os.path.splitext(args.input)
    splits = {
        f"{base}-train{ext}": train_data,
        f"{base}-validate{ext}": val_data,
        f"{base}-test{ext}": test_data,
    }

    for path, split in splits.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
        print(f"Saved {len(split)} samples -> {path}")


# ===========================================================================
# Subcommand dispatch
# ===========================================================================

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] not in ("build", "split"):
        print("Usage: python -m mm2pipeline_data.dataset {build|split} [options]")
        sys.exit(2)
    sub, rest = argv[0], argv[1:]
    if sub == "build":
        main_build(rest)
    else:
        main_split(rest)


if __name__ == "__main__":
    main()
