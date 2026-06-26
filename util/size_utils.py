"""Helpers for working with variable-width level scenes.

Mixed-size training produces datasets whose scenes span a range of widths. These
helpers let the generation/evaluation scripts (a) discover that range from a
dataset, (b) snap a chosen width to something the UNet can actually denoise, and
(c) draw a random width inside the range. Keeping them here (pure json/math/
random) means run_diffusion and evaluate_caption_adherence can import
them without pulling in the heavier training stack.
"""
import json
import math
import random


def dataset_width_range(json_path):
    """Return (min_width, max_width) across the scenes in a level dataset JSON.

    Width is the column count of each scene (len(scene[0])). Returns None
    when the file has no scene entries (e.g. a RandomTest captions-only file), so
    callers can fall back to an explicit range or a different reference dataset.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    widths = [
        len(item["scene"][0])
        for item in data
        if isinstance(item, dict) and item.get("scene")
    ]
    if not widths:
        return None
    return min(widths), max(widths)


def unet_width_factor(unet):
    """Smallest width multiple a UNet can denoise: 2 ** (number of downsamples).

    Each down block halves the spatial size except the last, so the input width
    must be divisible by 2 ** (len(block_out_channels) - 1). Random widths are
    snapped to this so generation does not crash on a shape mismatch.
    """
    return 2 ** (len(unet.config.block_out_channels) - 1)


def sample_random_width(min_width, max_width, factor=1, rng=random):
    """Draw a random width in [min_width, max_width] that is a multiple of factor.

    The endpoints coming from real training data are already multiples of
    factor (otherwise the model could not have trained on them), so the snapped
    range [ceil(min/factor), floor(max/factor)] is non-empty. rng is any
    object with a randint method (defaults to the random module); pass a
    seeded random.Random for reproducible width sequences.
    """
    lo = math.ceil(min_width / factor)
    hi = max_width // factor
    if hi < lo:
        hi = lo
    return rng.randint(lo, hi) * factor


def round_up_to_multiple(value, factor):
    """Smallest multiple of factor that is >= value (factor >= 1).

    Used to snap a bucket's padded width/height up to something the UNet can
    actually denoise (see unet_width_factor). factor <= 1 is a no-op.
    """
    if factor <= 1:
        return int(value)
    return int(math.ceil(value / factor) * factor)


def compute_size_buckets(sizes, num_buckets, factor=1):
    """Group variable-size levels into num_buckets size buckets of comparable
    population, returning the padded shape each bucket pads its members up to.

    Implements an expanding-rectangle scheme: a rectangle anchored at the origin
    grows toward the corner (max_w, max_h), capturing levels in the order they
    first fall inside it, split so each bucket holds ~the same number of levels.
    A level's capture order is key = max(w / max_w, h / max_h) -- normalizing
    by each axis (rather than a 45-degree square) keeps wide-short and tall-narrow
    levels balanced instead of bucketing almost purely by width.

    Args:
        sizes: list of (w, h) per level (width = columns, height = rows).
        num_buckets: desired number of buckets (>= 1). Fewer may be returned when
            neighbouring buckets round to identical padded shapes.
        factor: pad each bucket dimension up to a multiple of this (the UNet
            spatial divisor); 1 leaves the dimensions untouched.

    Returns:
        list of dicts {"pad_w", "pad_h", "indices"} ordered small -> large.
        indices are positions into sizes; every level appears in exactly
        one bucket and fits within its bucket's (pad_w, pad_h).
    """
    n = len(sizes)
    if n == 0:
        return []
    num_buckets = max(1, min(num_buckets, n))

    max_w = max(w for w, _ in sizes)
    max_h = max(h for _, h in sizes)

    def capture_key(i):
        w, h = sizes[i]
        return max(w / max_w, h / max_h)

    order = sorted(range(n), key=capture_key)

    buckets = []
    for b in range(num_buckets):
        # Contiguous, near-equal-count slices of the capture order.
        lo = (b * n) // num_buckets
        hi = ((b + 1) * n) // num_buckets
        group = order[lo:hi]
        if not group:
            continue
        pad_w = round_up_to_multiple(max(sizes[i][0] for i in group), factor)
        pad_h = round_up_to_multiple(max(sizes[i][1] for i in group), factor)
        buckets.append({"pad_w": pad_w, "pad_h": pad_h, "indices": group})

    # Merge neighbours that rounded to the same padded shape so we don't emit
    # redundant bucket sizes (e.g. two adjacent groups both snapping to 32-wide).
    merged = []
    for bucket in buckets:
        if merged and merged[-1]["pad_w"] == bucket["pad_w"] and merged[-1]["pad_h"] == bucket["pad_h"]:
            merged[-1]["indices"].extend(bucket["indices"])
        else:
            merged.append(bucket)
    return merged


def level_content_box(rows, empty_chars="-@", fill=None):
    """Return one ASCII level's CONTENT bounding box as a list of equal-width strings.

    Sky padding is excluded on all four sides: after stripping trailing newlines,
    the characters in empty_chars are treated as blank and the tightest box
    that still contains every non-empty tile is cut out. So a level whose tiles
    only span, say, 100 columns and 14 rows comes back as 14 strings of length
    100, not the padded canvas size.

    Ragged rows (shorter than the box) are right-filled with fill (defaults to
    the first empty char) so every returned row is exactly the box width. Returns
    [] for a level with no content at all.

    In datasets/MM.json the air tile is - and @ is the out-of-bounds
    null, so the default treats both as empty.
    """
    rows = [r.rstrip("\r\n") for r in rows]
    empty = set(empty_chars)
    if fill is None:
        fill = empty_chars[0] if empty_chars else " "

    def row_has_content(r):
        return any(ch not in empty for ch in r)

    # Vertical extent: first and last rows that hold any non-empty tile.
    top = next((i for i, r in enumerate(rows) if row_has_content(r)), None)
    if top is None:
        return []
    bottom = next(i for i in range(len(rows) - 1, -1, -1) if row_has_content(rows[i]))
    content_rows = rows[top: bottom + 1]

    # Horizontal extent: leftmost and rightmost columns holding a non-empty tile
    # across the content rows.
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
    """Return (width, height) of one ASCII level's content bounding box.

    Thin wrapper over level_content_box. Returns (0, 0) for a level with
    no content at all.
    """
    box = level_content_box(rows, empty_chars=empty_chars)
    if not box:
        return 0, 0
    return len(box[0]), len(box)
