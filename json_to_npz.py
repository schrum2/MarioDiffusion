"""Numpy bundle format for level datasets.

A level dataset is normally a JSON list of dicts, each with a "scene" (a list-of-lists
of small tile ids) plus caption/prompt metadata. Stored that way, every tile is a Python
int (~28 bytes) inside nested lists, so a dataset costs ~30x its real size in RAM, every
run re-parses multi-GB of text with json.load, and (on Windows spawn) every DataLoader
worker gets its own pickled copy.

This module stores the scenes as one contiguous uint8 numpy array (tile ids are 0..255)
that can be memory-mapped -- near-instant load, no RAM spike, shared across workers -- and
keeps the caption metadata in a small JSON sidecar. LevelDataset uses load_any()/save_bundle()
here to transparently cache and reuse the numpy form; this file's __main__ is the one-time
CLI preprocessor.

Bundle layout for a source `datasets/foo.json` (base = `datasets/foo`):

    foo.scenes.npy    scenes. Fixed-shape: (N, H, W) uint8. Ragged: flat 1-D uint8 buffer.
    foo.offsets.npy   ragged only: (N+1,) int64, scene i = flat[offsets[i]:offsets[i+1]].
    foo.shapes.npy    ragged only: (N, 2) int32, the (H, W) to reshape each scene to.
    foo.meta.json     header (format/version/shape/dtype/source) + per-item metadata
                      (every dict key except "scene"), one entry per scene, in source order.

A single-file `.npz` (scenes + offsets/shapes + a "meta" json string) is also accepted as
input for portability; the memmap-friendly multi-file bundle is what we generate by default.
"""

import argparse
import json
import os

import numpy as np

BUNDLE_VERSION = 1
# uint8 covers every tileset we have (max id ~41); fall back to int16 only if a dataset
# ever exceeds 255 tile ids.
_MAX_UINT8 = 255


# --------------------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------------------

def bundle_base(path):
    """Return the bundle base (path without extension/suffix) for any accepted input path.

    `datasets/foo.json` -> `datasets/foo`; `datasets/foo.scenes.npy` -> `datasets/foo`;
    `datasets/foo.npz` -> `datasets/foo`; `datasets/foo` -> `datasets/foo`.
    """
    for suffix in (".scenes.npy", ".meta.json", ".offsets.npy", ".shapes.npy"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    root, ext = os.path.splitext(path)
    if ext.lower() in (".json", ".npy", ".npz"):
        return root
    return path


def _paths(base):
    return {
        "scenes": base + ".scenes.npy",
        "offsets": base + ".offsets.npy",
        "shapes": base + ".shapes.npy",
        "meta": base + ".meta.json",
    }


def bundle_exists(base):
    """True if the two always-present bundle files (scenes + meta) exist for this base."""
    p = _paths(base)
    return os.path.exists(p["scenes"]) and os.path.exists(p["meta"])


def _source_stat(source_path):
    """Small fingerprint of the source JSON used to detect a stale cache."""
    if source_path is None or not os.path.exists(source_path):
        return None
    st = os.stat(source_path)
    return {"path": os.path.abspath(source_path), "mtime": st.st_mtime, "size": st.st_size}


def bundle_fresh(base, source_path):
    """True if a bundle exists and was built from the current version of source_path.

    Compares the source file's (mtime, size) against the fingerprint stored in the bundle
    header, so an edited/regenerated JSON transparently invalidates the cache.
    """
    if not bundle_exists(base):
        return False
    try:
        with open(_paths(base)["meta"], "r") as f:
            header = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if header.get("version") != BUNDLE_VERSION:
        return False
    stored = header.get("source")
    current = _source_stat(source_path)
    if stored is None or current is None:
        # No fingerprint to compare (e.g. bundle built without a source): treat as fresh
        # only if we also have no source to check against.
        return current is None
    return stored.get("size") == current.get("size") and stored.get("mtime") == current.get("mtime")


# --------------------------------------------------------------------------------------
# Conversion: list-of-dicts  <->  numpy arrays
# --------------------------------------------------------------------------------------

def _all_have_scene(data):
    return all(isinstance(item, dict) and item.get("scene") is not None for item in data)


def build_arrays(data):
    """Pack the "scene" of every item into numpy arrays; return (arrays, meta_items, header).

    `data` is a list of dicts each with a "scene" (list-of-lists or ndarray of tile ids).
    Fixed-shape datasets pack into a single (N, H, W) array; ragged datasets pack into a
    flat 1-D buffer with offsets + per-item shapes. `meta_items` is the same list with the
    "scene" key removed from each dict (captions/prompt/etc., preserved in source order).

    Raises ValueError if any item lacks a scene -- caption-only datasets are not cached.
    """
    if not _all_have_scene(data):
        raise ValueError("build_arrays requires every item to have a non-empty 'scene'")

    scenes = [np.asarray(item["scene"]) for item in data]
    max_id = max(int(s.max()) for s in scenes) if scenes else 0
    dtype = np.uint8 if max_id <= _MAX_UINT8 else np.int16

    shapes = [s.shape for s in scenes]
    fixed = len(set(shapes)) == 1
    meta_items = [{k: v for k, v in item.items() if k != "scene"} for item in data]

    if fixed:
        packed = np.asarray(scenes, dtype=dtype)  # (N, H, W)
        arrays = {"scenes": packed}
        header = {
            "format": "fixed",
            "shape": list(packed.shape[1:]),
        }
    else:
        flat = np.concatenate([s.reshape(-1).astype(dtype, copy=False) for s in scenes])
        lengths = np.array([s.size for s in scenes], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        shape_arr = np.array(shapes, dtype=np.int32)  # (N, 2)
        arrays = {"scenes": flat, "offsets": offsets, "shapes": shape_arr}
        header = {"format": "ragged", "shape": None}

    header.update({
        "version": BUNDLE_VERSION,
        "num_items": len(data),
        "dtype": np.dtype(dtype).name,
    })
    return arrays, meta_items, header


def _rebuild_data(arrays, meta_items, header):
    """Rebuild the list-of-dicts, with each "scene" a (memmap-backed) view of the arrays."""
    scenes = arrays["scenes"]
    if header["format"] == "fixed":
        data = []
        for i, meta in enumerate(meta_items):
            item = dict(meta)
            item["scene"] = scenes[i]  # (H, W) view; no copy
            data.append(item)
        return data

    offsets = arrays["offsets"]
    shapes = arrays["shapes"]
    data = []
    for i, meta in enumerate(meta_items):
        item = dict(meta)
        h, w = int(shapes[i][0]), int(shapes[i][1])
        item["scene"] = scenes[offsets[i]:offsets[i + 1]].reshape(h, w)  # view; no copy
        data.append(item)
    return data


# --------------------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------------------

def save_bundle(base, data, source_path=None):
    """Convert `data` and write the memmap-friendly bundle at `base`. Returns the base.

    `source_path`, if given, is fingerprinted into the header so bundle_fresh() can detect
    when the source JSON later changes.
    """
    arrays, meta_items, header = build_arrays(data)
    header["source"] = _source_stat(source_path)
    p = _paths(base)

    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    np.save(p["scenes"], arrays["scenes"])
    if header["format"] == "ragged":
        np.save(p["offsets"], arrays["offsets"])
        np.save(p["shapes"], arrays["shapes"])
    else:
        # Remove any stale ragged sidecars from a previous build of a differently-shaped source.
        for key in ("offsets", "shapes"):
            if os.path.exists(p[key]):
                os.remove(p[key])
    with open(p["meta"], "w") as f:
        json.dump({**header, "items": meta_items}, f)
    return base


def load_bundle(base, mmap=True):
    """Load a bundle at `base` into a list-of-dicts, scenes backed by memmap views if mmap."""
    p = _paths(base)
    with open(p["meta"], "r") as f:
        header = json.load(f)
    meta_items = header["items"]
    mmap_mode = "r" if mmap else None
    arrays = {"scenes": np.load(p["scenes"], mmap_mode=mmap_mode)}
    if header["format"] == "ragged":
        arrays["offsets"] = np.load(p["offsets"], mmap_mode=mmap_mode)
        arrays["shapes"] = np.load(p["shapes"], mmap_mode=mmap_mode)
    return _rebuild_data(arrays, meta_items, header)


def save_npz(path, data, source_path=None, compressed=False):
    """Write a single portable .npz (scenes + offsets/shapes + meta json string).

    Convenient one-file form; note a .npz cannot be memory-mapped, so load_npz reads it
    fully into RAM. The multi-file bundle (save_bundle) is preferred for large datasets.
    """
    arrays, meta_items, header = build_arrays(data)
    header["source"] = _source_stat(source_path)
    payload = dict(arrays)
    payload["meta"] = np.array(json.dumps({**header, "items": meta_items}))
    saver = np.savez_compressed if compressed else np.savez
    saver(path, **payload)


def load_npz(path):
    """Load a single-file .npz produced by save_npz into a list-of-dicts."""
    with np.load(path, allow_pickle=False) as npz:
        header = json.loads(str(npz["meta"]))
        arrays = {"scenes": npz["scenes"]}
        if header["format"] == "ragged":
            arrays["offsets"] = npz["offsets"]
            arrays["shapes"] = npz["shapes"]
        return _rebuild_data(arrays, header["items"], header)


def load_any(path, mmap=True):
    """Load level data from a .json, .npy/bundle, or .npz path into a list-of-dicts.

    - .json: parsed as-is (scenes left as Python lists; the caller converts/caches).
    - .npz: single-file bundle (load_npz).
    - .npy / bundle base / .scenes.npy: multi-file bundle (load_bundle, memmapped if mmap).
    """
    lower = path.lower()
    if lower.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    if lower.endswith(".npz"):
        return load_npz(path)
    return load_bundle(bundle_base(path), mmap=mmap)


# --------------------------------------------------------------------------------------
# CLI: one-time preprocessing
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess a level JSON dataset into a numpy bundle.")
    parser.add_argument("input", help="Source dataset .json")
    parser.add_argument("--out", default=None, help="Output bundle base (default: input path minus .json)")
    parser.add_argument("--npz", action="store_true", help="Also write a single-file .npz alongside the bundle")
    parser.add_argument("--compressed", action="store_true", help="Compress the .npz (implies --npz; not memmappable)")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a fresh bundle already exists")
    args = parser.parse_args()

    base = args.out or bundle_base(args.input)
    if not args.force and bundle_fresh(base, args.input):
        print(f"Bundle already fresh at {base}.* -- use --force to rebuild.")
        return

    print(f"Loading {args.input} ...")
    with open(args.input, "r") as f:
        data = json.load(f)
    print(f"  {len(data)} items")

    save_bundle(base, data, source_path=args.input)
    print(f"Wrote bundle: {base}.scenes.npy (+ .meta.json)")

    if args.npz or args.compressed:
        npz_path = base + ".npz"
        save_npz(npz_path, data, source_path=args.input, compressed=args.compressed)
        print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
