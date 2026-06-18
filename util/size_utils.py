"""Helpers for working with variable-width level scenes.

Mixed-size training produces datasets whose scenes span a range of widths. These
helpers let the generation/evaluation scripts (a) discover that range from a
dataset, (b) snap a chosen width to something the UNet can actually denoise, and
(c) draw a random width inside the range. Keeping them here (pure ``json``/``math``/
``random``) means ``run_diffusion`` and ``evaluate_caption_adherence`` can import
them without pulling in the heavier training stack.
"""
import json
import math
import random


def dataset_width_range(json_path):
    """Return ``(min_width, max_width)`` across the scenes in a level dataset JSON.

    Width is the column count of each scene (``len(scene[0])``). Returns ``None``
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
    """Smallest width multiple a UNet can denoise: ``2 ** (number of downsamples)``.

    Each down block halves the spatial size except the last, so the input width
    must be divisible by ``2 ** (len(block_out_channels) - 1)``. Random widths are
    snapped to this so generation does not crash on a shape mismatch.
    """
    return 2 ** (len(unet.config.block_out_channels) - 1)


def sample_random_width(min_width, max_width, factor=1, rng=random):
    """Draw a random width in ``[min_width, max_width]`` that is a multiple of ``factor``.

    The endpoints coming from real training data are already multiples of
    ``factor`` (otherwise the model could not have trained on them), so the snapped
    range ``[ceil(min/factor), floor(max/factor)]`` is non-empty. ``rng`` is any
    object with a ``randint`` method (defaults to the ``random`` module); pass a
    seeded ``random.Random`` for reproducible width sequences.
    """
    lo = math.ceil(min_width / factor)
    hi = max_width // factor
    if hi < lo:
        hi = lo
    return rng.randint(lo, hi) * factor
