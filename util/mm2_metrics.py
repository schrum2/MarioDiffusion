"""Metrics over Mario Maker 2 level scenes (68-tile mm2_tileset_we encoding).

1. Broken structure detection. mm2pipeline_data.ascii paints every object as a w x h
   block of one glyph, and mm2pipeline_data.ascii's COALESCE_POLICY records the
   footprint each object should have. A blob that doesn't form its footprint is
   visibly wrong -- the repair pass just stamps it into enough valid objects to
   cover it (a scattered clump of '!' becomes several 2x2 Boom Booms).
   count_structures() applies those footprint rules to count, per feature, the
   instances in a scene and how many are broken.

2. Average Minimum Edit Distance (AMED), self and vs-real, from the previous
   MarioDiffusion paper (util/metrics.py). MM2 levels come in bucketed sizes, so
   these group by shape and only compare like-sized levels.

3. Maximum edit distance within a group of scenes generated from different
   captions of the same source scene -- high means the captions led to very
   different levels.
"""
import os
import sys
import torch

# Repo root on the path so the converter modules resolve from the util dir.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm2pipeline_data.tiles import OBJ_META, CHAR_TO_NAME
from mm2pipeline_data.ascii import (COALESCE_POLICY, PLATFORM_MIN_SIZE,
                               _connected_components, _runs)

_FIXED, _BBOX, _MUSHROOM, _HRUN, _VRUN, _PIPE = (
    "fixed", "bbox", "mushroom", "hrun", "vrun", "pipe")

# Footprint corrections to COALESCE_POLICY from component-shape stats over the
# real training data: interior Banzai Bill blobs are 4x4, not the assumed 2x2.
FOOTPRINT_OVERRIDES = {
    "Banzai Bill": (4, 4),
}

# Footprints too variable to judge (Bowser Jr. 1x1/2x2, Clown Car many sizes,
# lone Goomba's Shoe common). Totals reported, nothing flagged broken.
UNCHECKED_FEATURES = {"Bowser Jr.", "Bowser Jr", "Clown Car", "Goomba's Shoe"}

# Blocks that can hold an item; an item glyph painted above one is its contents.
CONTAINER_CHARS = {"B", "?", "h"}

# Solid terrain/block glyphs plus '_' padding. Ground is painted last, so an
# object embedded in terrain (buried saw, pipe in a hillside) loses those cells
# yet still looks fine in-game. Such cells (and off-scene ones) count as
# wildcards: a footprint window is complete if its missing cells are wildcards.
WILDCARD_CHARS = {"#", "H", "B", "?", "N", "I", "O", ".", "<", "^", "=", "k", "T", "_"}

# Track hazards overlap arbitrary objects, so for them ANY overpainted cell is a
# wildcard, not just terrain.
TRACK_FEATURES = {"Saw", "Skewer", "Swinging Claw"}


def _build_feature_policies():
    """glyph -> (canonical name, coalesce policy) for every multi-tile object.
    Glyph-sharing aliases (Yoshi's Egg / Goomba's Shoe) collapse to the glyph's
    canonical CHAR_TO_NAME entry."""
    feats = {}
    for name, policy in COALESCE_POLICY.items():
        char = OBJ_META.get(name, (None,))[0]
        if char is None or CHAR_TO_NAME.get(char) != name:
            continue
        if name in FOOTPRINT_OVERRIDES:
            policy = (_FIXED,) + FOOTPRINT_OVERRIDES[name]
        feats[char] = (name, policy)
    return feats


FEATURE_POLICIES = _build_feature_policies()


def _is_full_rect(comp):
    cols = [c for c, _ in comp]
    rows = [r for _, r in comp]
    w = max(cols) - min(cols) + 1
    h = max(rows) - min(rows) + 1
    return len(comp) == w * h, w, h


def _count_fixed(comp, fw, fh, scene_w, scene_h, wildcards):
    """Tile the bounding box into fw x fh stamps like coalesce does. Each stamp
    with a cell is one instance, valid when some placement of the full fw x fh
    window covers all its cells with every other window cell explainable (part
    of the component, a wildcard, or off-scene); otherwise it's a fragment.

    A big-form enemy paints a 2fw x 2fh block, which splits into four valid
    stamps -- so the total counts one big enemy as four normal ones.
    """
    cells = set(comp)

    def explainable(c, r):
        if c < 0 or c >= scene_w or r < 0 or r >= scene_h:
            return True
        return (c, r) in cells or (c, r) in wildcards

    def fits_window(stamp_cells):
        sc0 = max(c for c, _ in stamp_cells) - fw + 1
        sc1 = min(c for c, _ in stamp_cells)
        sr0 = max(r for _, r in stamp_cells) - fh + 1
        sr1 = min(r for _, r in stamp_cells)
        for wc in range(sc0, sc1 + 1):
            for wr in range(sr0, sr1 + 1):
                if all(explainable(c, r)
                       for c in range(wc, wc + fw)
                       for r in range(wr, wr + fh)):
                    return True
        return False

    cols = [c for c, _ in comp]
    rows = [r for _, r in comp]
    c0, c1, r0, r1 = min(cols), max(cols), min(rows), max(rows)
    total = broken = 0
    cr = r0
    while cr <= r1:
        cc = c0
        while cc <= c1:
            w = min(fw, c1 - cc + 1)
            h = min(fh, r1 - cr + 1)
            stamp_cells = [(c, r) for c in range(cc, cc + w)
                           for r in range(cr, cr + h) if (c, r) in cells]
            if stamp_cells:
                total += 1
                if not fits_window(stamp_cells):
                    broken += 1
            cc += fw
        cr += fh
    return total, broken


def _count_pipe(comp, fill_cells, scene_w, scene_h):
    """A pipe is a full even-width rectangle at least 2 long (horizontal pipes
    are the transpose); a 1-wide column flush against a left/right edge is a
    cropped pipe. Cells painted over by a piranha at the mouth, an item leaving
    the pipe, or embedding terrain are filled back in first. Else broken."""
    cols = [c for c, _ in comp]
    rows = [r for _, r in comp]
    c0, c1, r0, r1 = min(cols), max(cols), min(rows), max(rows)
    cells = set(comp)
    cells |= {(c, r) for (c, r) in fill_cells
              if c0 <= c <= c1 and r0 <= r <= r1}
    w = c1 - c0 + 1
    h = r1 - r0 + 1
    full = len(cells) == w * h
    valid = full and (
        (w % 2 == 0 and h >= 2)
        or (h % 2 == 0 and w >= 2)
        or (w == 1 and h >= 2 and (c0 == 0 or c1 == scene_w - 1))
        or (h == 1 and w >= 2 and (r0 == 0 or r1 == scene_h - 1))
    )
    return 1, 0 if valid else 1


def _count_mushroom(comp, scene_h, occupied):
    """Mushroom platforms are a wide cap over a centred stem; stacked ones share
    a blob. Caps are found as in coalesce(): a row of >= 2 cells with <= 1 cell
    above. No cap is broken -- unless the blob reaches the scene top (cap
    cropped) or something painted over the background pass covers the cap."""
    rows_by = {}
    for c, r in comp:
        rows_by.setdefault(r, []).append(c)
    caps = [r for r, cs in rows_by.items()
            if len(cs) >= 2 and len(rows_by.get(r + 1, ())) <= 1]
    if not caps:
        top = max(rows_by)
        cap_occluded = any((c, top + 1) in occupied for c in rows_by[top])
        if top == scene_h - 1 or cap_occluded:
            return 1, 0
        return 1, 1
    return len(caps), 0


def _count_semisolid(comp, occupied):
    """Semisolid platforms are background boxes that everything painted after
    punches holes in, so their real shape is unrecoverable. Only unplaceable
    specks count as broken: blobs under the 3-tile minimum with nothing
    adjacent that could be hiding the rest of the platform."""
    if len(comp) >= PLATFORM_MIN_SIZE:
        return 1, 0
    occluded = any((c + dc, r + dr) in occupied
                   for c, r in comp
                   for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    return 1, 0 if occluded else 1


def _count_blaster(cells, scene_h, wildcards):
    """Bullet Bill Blasters are 1-wide columns at least 2 tall (head + base),
    counted per column run. A lone cell at the top/bottom edge is cropped, and
    one with terrain directly above or below has its base buried -- not broken."""
    by_col = {}
    for c, r in cells:
        by_col.setdefault(c, []).append(r)
    total = broken = 0
    for c, crows in by_col.items():
        for start, length in _runs(crows):
            total += 1
            if length >= 2 or start == 0 or start + length == scene_h:
                continue
            if (c, start - 1) in wildcards or (c, start + length) in wildcards:
                continue
            broken += 1
    return total, broken


def count_structures(scene, id_to_char):
    """Count multi-tile structures in one scene.

    Returns {feature name: {"total": n, "broken": m}} covering every feature
    with a defined footprint that appears in the scene. Coordinates are
    (col, row) with row 0 at the bottom, matching mm2pipeline_data.ascii.
    """
    scene_h = len(scene)
    scene_w = len(scene[0]) if scene_h else 0
    cells_by_char = {}
    container_cells = set()
    wildcards = set()
    occupied = set()
    for r, row in enumerate(scene):
        for c, tile in enumerate(row):
            ch = id_to_char.get(tile)
            if ch is None or ch == " ":
                continue
            pos = (c, scene_h - 1 - r)
            occupied.add(pos)
            if ch in FEATURE_POLICIES:
                cells_by_char.setdefault(ch, set()).add(pos)
            if ch in CONTAINER_CHARS:
                container_cells.add(pos)
            if ch in WILDCARD_CHARS:
                wildcards.add(pos)

    counts = {}
    for ch, cells in cells_by_char.items():
        name, policy = FEATURE_POLICIES[ch]
        kind = policy[0]
        total = broken = 0
        # Cells this feature doesn't own, for occlusion checks (a feature can't
        # hide behind its own glyph).
        occupied_other = occupied - cells

        if name in UNCHECKED_FEATURES:
            # Lone glyphs sitting on an item block are the block's contents.
            free = {cell for cell in cells
                    if (cell[0], cell[1] - 1) not in container_cells}
            total = len(_connected_components(free)) if free else 0
        elif name == "Bullet Bill Blaster":
            total, broken = _count_blaster(cells, scene_h, wildcards)
        elif name == "Semisolid Platform":
            for comp in _connected_components(cells):
                t, b = _count_semisolid(comp, occupied_other)
                total += t
                broken += b
        elif kind == _FIXED:
            # Track hazards overlap anything, so any overpaint explains a cell.
            wild = occupied_other | wildcards if name in TRACK_FEATURES else wildcards
            for comp in _connected_components(cells):
                t, b = _count_fixed(comp, policy[1], policy[2], scene_w, scene_h,
                                    wild)
                total += t
                broken += b
        elif kind == _PIPE:
            for comp in _connected_components(cells):
                t, b = _count_pipe(comp, occupied_other, scene_w, scene_h)
                total += t
                broken += b
        elif kind == _MUSHROOM:
            for comp in _connected_components(cells):
                t, b = _count_mushroom(comp, scene_h, occupied_other)
                total += t
                broken += b
        else:
            # HRUN/VRUN/bbox features have no wrong shape: count only, never broken.
            if kind == _VRUN or kind == _HRUN:
                axis = 0 if kind == _HRUN else 1
                by = {}
                for c, r in cells:
                    by.setdefault((c, r)[1 - axis], []).append((c, r)[axis])
                total = sum(len(list(_runs(v))) for v in by.values())
            else:
                total = len(_connected_components(cells))

        if total or broken:
            counts[name] = {"total": total, "broken": broken}
    return counts


def broken_structure_report(scenes, id_to_char):
    """Aggregate count_structures over a list of scenes.

    Returns {feature: {"total", "broken", "scenes_with_feature",
    "scenes_with_broken"}} plus a "scenes_with_any_broken" count under the
    reserved key "_overall"."""
    report = {}
    scenes_with_any = 0
    for scene in scenes:
        counts = count_structures(scene, id_to_char)
        any_broken = False
        for name, c in counts.items():
            entry = report.setdefault(name, {
                "total": 0, "broken": 0,
                "scenes_with_feature": 0, "scenes_with_broken": 0})
            entry["total"] += c["total"]
            entry["broken"] += c["broken"]
            entry["scenes_with_feature"] += 1
            if c["broken"]:
                entry["scenes_with_broken"] += 1
                any_broken = True
        if any_broken:
            scenes_with_any += 1
    report["_overall"] = {
        "num_scenes": len(scenes),
        "scenes_with_any_broken": scenes_with_any,
    }
    return report


# ---------------------------------------------------------------------------
# Edit distance metrics
# ---------------------------------------------------------------------------

def _group_by_shape(levels):
    groups = {}
    for i, level in enumerate(levels):
        shape = (len(level), len(level[0]) if level else 0)
        groups.setdefault(shape, []).append(i)
    return groups


def average_min_edit_distance(level_collection, use_gpu=True):
    """AMED_self: averaged tile-wise distance from each level to its nearest
    same-shape neighbour. A level whose shape appears once is skipped.
    Returns (average, num_compared, num_skipped)."""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    total = 0.0
    compared = 0
    skipped = 0
    for shape, indices in _group_by_shape(level_collection).items():
        if len(indices) < 2:
            skipped += len(indices)
            continue
        levels = torch.tensor([level_collection[i] for i in indices],
                              dtype=torch.int16).to(device)
        for i in range(len(indices)):
            others = torch.cat([levels[:i], levels[i + 1:]])
            total += (others != levels[i]).sum(dim=(1, 2)).min().item()
            compared += 1
    if skipped:
        print(f"Warning: {skipped} level(s) had no same-shape partner and were skipped in AMED_self")
    return (total / compared if compared else 0.0), compared, skipped


def average_min_edit_distance_from_real(generated_levels, real_levels, use_gpu=True):
    """AMED_real: each generated level's distance to the nearest same-shape real
    level; generated levels with no same-shape real level are skipped.
    Returns (average, perfect_matches, num_compared, num_skipped)."""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    real_by_shape = {
        shape: torch.tensor([real_levels[i] for i in idx], dtype=torch.int16).to(device)
        for shape, idx in _group_by_shape(real_levels).items()
    }
    total = 0.0
    perfect = 0
    compared = 0
    skipped = 0
    for level in generated_levels:
        shape = (len(level), len(level[0]) if level else 0)
        real = real_by_shape.get(shape)
        if real is None:
            skipped += 1
            continue
        gen = torch.tensor(level, dtype=torch.int16).to(device)
        min_dist = (real != gen).sum(dim=(1, 2)).min().item()
        if min_dist == 0:
            perfect += 1
        total += min_dist
        compared += 1
    if skipped:
        print(f"Warning: {skipped} generated level(s) had no same-shape real level and were skipped in AMED_real")
    return (total / compared if compared else 0.0), perfect, compared, skipped


def max_edit_distance(levels, use_gpu=True):
    """Largest pairwise tile-wise distance within one group of levels (e.g.
    scenes from captions of the same source scene). Mismatched-shape pairs are
    ignored; returns None if no pair could be compared."""
    if len(levels) < 2:
        return None
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    best = None
    for indices in _group_by_shape(levels).values():
        if len(indices) < 2:
            continue
        group = torch.tensor([levels[i] for i in indices], dtype=torch.int16).to(device)
        for i in range(len(indices) - 1):
            dist = (group[i + 1:] != group[i]).sum(dim=(1, 2)).max().item()
            if best is None or dist > best:
                best = dist
    return best


def source_group_key(entry):
    """Which source scene a generated entry's prompt came from, from the
    metadata evaluate_caption_adherence.py writes. None when absent."""
    if "source_name" in entry and entry["source_name"] is not None:
        return entry["source_name"]
    if "source_index" in entry and entry["source_index"] is not None:
        return entry["source_index"]
    return None


if __name__ == "__main__":
    # Issue #37's Boom Boom example: a ragged blob should count as several
    # broken 2x2 Boom Booms.
    from Game_MM2.MarioMaker_create_ascii_captions import build_id_to_char
    import util.common_settings as common_settings

    id_to_char = build_id_to_char(common_settings.MM2_TILESET)
    char_to_id = {v: k for k, v in id_to_char.items()}
    rows = ["       ",
            " !!-!- ",
            " -!!-- ",
            " !!!!! ",
            "       "]
    scene = [[char_to_id['!' if ch == '!' else ' '] for ch in row] for row in rows]
    print(count_structures(scene, id_to_char))
