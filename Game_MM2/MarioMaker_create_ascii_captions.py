import json
import os
import sys
import argparse

# Only for running this file directly to debug it; imports already find the root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Tags used to describe tile properties rather than identity; everything in a
# tile's tag list other than these is treated as part of its name.
PROPERTY_TAGS = {
    "passable", "solid", "empty", "air", "breakable", "collectable", "enemy",
    "damaging", "hazard", "moving", "flying", "projectile", "explosive",
    "shooter", "power-up", "style ride", "platform",
    "interactive", "climbable", "togglable", "slippery", "falling", "warp",
    "door", "vehicle",
}

# Tiles that represent empty space and should never appear in a caption.
EMPTY_TAGS = {"empty", "air"}

# Level metadata fields to fold into the caption, paired with the word that
# turns the raw value into a phrase. level_name is left out on purpose - it names
# the source level, not its contents.
CAPTION_METADATA_FIELDS = [
    ("gamestyle", "style"),
    ("theme", "theme"),
    ("difficulty", "difficulty"),
]

# A same-tile contiguous region at least this big gets called a blob.
BLOB_THRESHOLD = 10

# A terrain region smaller than this is left to the tile counts.
TERRAIN_MIN_REGION = 4

# A region on the bottom row at least this wide is the ground, not a structure.
GROUND_WIDTH_FRACTION = 0.4

# Surface heights within this much of each other read as level ground.
SURFACE_FLAT_TOLERANCE = 2

# A ledge this much of the scene wide is doing the job of a floor.
WIDE_LEDGE_FRACTION = 0.6

# A region packed at least this solid is a block rather than a room.
SOLID_FILL_FRACTION = 0.7

# Surfaces that climb steadily one way. A region shaped like this is a staircase.
SLOPED_SURFACES = ("rising to the right", "sloping down to the right")

# A staircase needs at least this much run and rise.
STAIRCASE_MIN_SPAN = 3

# Ground covering this much of the scene gets called out as bulk terrain.
GROUND_BULK_FRACTION = 0.6
GROUND_HEAVY_FRACTION = 0.35

# Gaps under terrain in this many columns read as a cave.
UNDERCUT_MIN_COLUMNS = 3

# Ground this many rows deep is a plateau, not the floor.
GROUND_TALL_ROWS = 6

# Tags that keep a tile out of the block set: it moves, hurts, or is picked up.
BLOCK_EXCLUDE_TAGS = {"damaging", "hazard", "enemy", "moving", "warp", "pipe",
                      "shooter", "collectable", "power-up"}

# Shape names for structures built out of blocks instead of ground.
BLOCK_SHAPE_NOUNS = {
    "ground": "platform",
    "wide ledge": "platform",
    "ledge": "row",
    "pillar": "column",
    "ground block": "wall",
    "staircase": "staircase",
    "hollow structure": "structure",
    "mound": "cluster",
}


def metadata_phrases(item):
    """Build caption phrases from an item's level metadata. Skips missing/empty
    values, the "Unknown" difficulty placeholder, and the "None" tag slot."""
    phrases = []
    for field, suffix in CAPTION_METADATA_FIELDS:
        value = item.get(field)
        if value in (None, "") or str(value).lower() == "unknown":
            continue
        phrases.append(f"{value} {suffix}")
    for tag in item.get("tags") or []:
        if tag and str(tag).lower() != "none":
            phrases.append(str(tag))
    return phrases


def build_id_to_char(tileset_path):
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tile_chars = sorted(data["tiles"].keys())
    if "_" not in tile_chars:
        tile_chars.append("_")
    return {idx: char for idx, char in enumerate(tile_chars)}


def get_char_names(tileset_path):
    """Map each tile char to a readable name read straight from its tag list, so
    names track whatever tileset is passed in (e.g. ["passable", "collectable",
    "coin"] -> "Coin")."""
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    char_names = {}
    for char, tags in data["tiles"].items():
        if not tags or any(t in EMPTY_TAGS for t in tags):
            continue
        # The name is the last non-property tag. Some tiles carry a category word
        # ahead of the name (Bowser is [..., "boss", "bowser"]), so taking the
        # last tag keeps every boss from being named "boss".
        name_tags = [t for t in tags if t not in PROPERTY_TAGS]
        name = name_tags[-1] if name_tags else tags[-1]
        char_names[char] = name.title()
    return char_names


def get_tile_categories(tileset_path):
    """Sort tile chars into (enemies, items, ground) by their tags. Enemies are
    tagged "enemy", items are collectables, ground is the terrain we call floor."""
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    enemy_chars, item_chars, ground_chars = set(), set(), set()
    for char, tags in data["tiles"].items():
        tagset = set(tags)
        if "enemy" in tagset:
            enemy_chars.add(char)
        if "collectable" in tagset:
            item_chars.add(char)
        if "ground" in tagset:
            ground_chars.add(char)
    return enemy_chars, item_chars, ground_chars


def get_block_chars(tileset_path):
    """
        Picks the block types that build structures the way ground does. Pipes and
        bridges stay out, since a footprint already counts them as one object.
        Returns the set of tile chars, without the ground itself.
    """
    from util.mm2_metrics import FEATURE_POLICIES

    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chars = set()
    for char, tags in data["tiles"].items():
        tagset = set(tags)
        if char in FEATURE_POLICIES or "ground" in tagset:
            continue
        if tagset & BLOCK_EXCLUDE_TAGS:
            continue
        if "solid" in tagset or tags[-1].endswith("block"):
            chars.add(char)
    return chars


def describe_quantity(count):
    # Coarse buckets like MarioDiffusion's, with a top "a ton of" tier and the
    # thresholds bumped up for the bigger Mario Maker scenes.
    if count == 1:
        return "one"
    if count == 2:
        return "two"
    if count < 6:
        return "a few"
    if count < 12:
        return "several"
    if count < 30:
        return "many"
    return "a ton of"


def pluralize(name):
    return name if name.endswith("s") else name + "s"


def count_phrase(count, name):
    if count <= 0:
        return None
    noun = pluralize(name) if count > 1 else name
    return f"{describe_quantity(count)} {noun}".capitalize()


def largest_blobs(scene, id_to_char):
    """
        Finds the largest contiguous same-tile region for each tile char by flood
        fill over 4-connected neighbours.
        Returns a dict mapping each char to the (row, col) positions of its region.
    """
    height = len(scene)
    width = len(scene[0]) if height else 0
    visited = set()
    biggest = {}
    for r in range(height):
        for c in range(width):
            if (r, c) in visited:
                continue
            char = id_to_char.get(scene[r][c])
            if char is None:
                continue
            stack = [(r, c)]
            blob = []
            while stack:
                y, x = stack.pop()
                if (y, x) in visited or not (0 <= y < height and 0 <= x < width):
                    continue
                if id_to_char.get(scene[y][x]) != char:
                    continue
                visited.add((y, x))
                blob.append((y, x))
                stack += [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
            if len(blob) > len(biggest.get(char, ())):
                biggest[char] = blob
    return biggest


def count_objects(scene, id_to_char):
    """
        Counts placed objects instead of occupied cells, so a 2x6 pipe counts once.
        Only the multi-tile types have a footprint to go by; the rest are one per cell.
        Returns a dict mapping tile char to the number of objects in the scene.
    """
    from util.mm2_metrics import count_structures, FEATURE_POLICIES

    counts = count_structures(scene, id_to_char)
    return {char: counts[name]["total"]
            for char, (name, _policy) in FEATURE_POLICIES.items()
            if name in counts}


def describe_ground(scene, id_to_char, ground_chars):
    if not ground_chars:
        return None
    if not any(id_to_char.get(t) in ground_chars for row in scene for t in row):
        return None

    bottom = [id_to_char.get(t) in ground_chars for t in scene[-1]]
    if all(bottom):
        return "Full ground floor"
    if not any(bottom):
        # Only the bottom row is ours; the region pass covers the rest.
        return None

    # Count the runs of missing ground along the bottom row.
    gaps = 0
    in_gap = False
    for is_ground in bottom:
        if not is_ground:
            gaps += not in_gap
            in_gap = True
        else:
            in_gap = False
    return f"Ground floor with {describe_quantity(gaps)} gap" + ("s" if gaps > 1 else "")


def terrain_regions(scene, id_to_char, terrain_chars):
    """
        Finds connected regions of terrain by flood fill over 4-connected
        neighbours. Every char in terrain_chars counts as the same material.
        Returns a list of (row, col) lists, largest region first.
    """
    height = len(scene)
    width = len(scene[0]) if height else 0
    visited = set()
    regions = []
    for r in range(height):
        for c in range(width):
            if (r, c) in visited or id_to_char.get(scene[r][c]) not in terrain_chars:
                continue
            stack = [(r, c)]
            region = []
            while stack:
                y, x = stack.pop()
                if (y, x) in visited or not (0 <= y < height and 0 <= x < width):
                    continue
                if id_to_char.get(scene[y][x]) not in terrain_chars:
                    continue
                visited.add((y, x))
                region.append((y, x))
                stack += [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
            regions.append(region)
    regions.sort(key=len, reverse=True)
    return regions


def surface_profile(cells):
    """
        Walks the top of a region, keeping the highest cell in each column.
        Returns one row number per column it covers, left to right.
    """
    tops = {}
    for r, c in cells:
        if c not in tops or r < tops[c]:
            tops[c] = r
    return [tops[c] for c in sorted(tops)]


def describe_surface(profile):
    """
        Names the shape of a region's top edge. Rows count down from the top of the
        scene, so a shrinking row number means the ground is climbing.
        Returns "flat", "rising to the right", "sloping down to the right" or "uneven".
    """
    if len(profile) < 2 or max(profile) - min(profile) <= SURFACE_FLAT_TOLERANCE:
        return "flat"
    if all(b <= a for a, b in zip(profile, profile[1:])):
        return "rising to the right"
    if all(b >= a for a, b in zip(profile, profile[1:])):
        return "sloping down to the right"
    return "uneven"


def undercut_columns(cells, height):
    """
        Counts the columns where the region has empty space directly below it,
        which is what a cave roof or a tunnel looks like from above.
        Returns the number of such columns.
    """
    occupied = set(cells)
    undercut = set()
    for r, c in cells:
        if r + 1 < height and (r + 1, c) not in occupied:
            undercut.add(c)
    return len(undercut)


def classify_terrain_region(cells, height, width):
    """
        Sorts a terrain region into a shape by its footprint, so a tall thin one is
        a pillar and a flat wide one is a ledge.
        Returns the shape name, or None when the region is too small to name.
    """
    if len(cells) < TERRAIN_MIN_REGION:
        return None
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    span_w = max(cols) - min(cols) + 1
    span_h = max(rows) - min(rows) + 1

    if max(rows) == height - 1 and span_w >= width * GROUND_WIDTH_FRACTION:
        return "ground"
    if span_h <= 2 and span_w >= 3:
        return "wide ledge" if span_w >= width * WIDE_LEDGE_FRACTION else "ledge"
    if span_w <= 2 and span_h >= 3:
        return "pillar"
    if len(cells) >= SOLID_FILL_FRACTION * span_w * span_h:
        return "ground block"
    # A top edge that climbs the whole way without turning back is a staircase.
    if span_w >= STAIRCASE_MIN_SPAN and span_h >= STAIRCASE_MIN_SPAN:
        if describe_surface(surface_profile(cells)) in SLOPED_SURFACES:
            return "staircase"
    # Big and mostly empty inside: walls around rooms or a maze, not a clump.
    if span_w >= 6 and span_h >= 4:
        return "hollow structure"
    return "mound"


def describe_terrain(scene, id_to_char, terrain_chars):
    """
        Describes the terrain above the bottom row, which the floor summary cannot
        reach. Shapes other than the ground are grouped so three ledges count once.
        Returns a list of (phrase, cells) pairs in the order they should be read.
    """
    height = len(scene)
    width = len(scene[0]) if height else 0
    phrases = []
    shapes = {}

    for region in terrain_regions(scene, id_to_char, terrain_chars):
        kind = classify_terrain_region(region, height, width)
        if kind is None:
            continue
        if kind == "ground":
            rows = [r for r, _ in region]
            surface = describe_surface(surface_profile(region))
            # Flat ground is already covered by the floor summary.
            if surface == "uneven":
                phrases.append(("Uneven ground", region))
            elif surface != "flat":
                phrases.append((f"Ground {surface}", region))
            if len(region) >= GROUND_BULK_FRACTION * height * width:
                phrases.append(("The ground fills most of the scene", region))
            elif len(region) >= GROUND_HEAVY_FRACTION * height * width:
                phrases.append(("The ground fills much of the scene", region))
            elif surface == "flat" and max(rows) - min(rows) + 1 >= GROUND_TALL_ROWS:
                # A deep flat mass is a plateau the floor summary misses.
                phrases.append(("Raised flat ground", region))
            if undercut_columns(region, height) >= UNDERCUT_MIN_COLUMNS:
                phrases.append(("Open space beneath the ground", region))
        else:
            count, cells = shapes.get(kind, (0, []))
            shapes[kind] = (count + 1, cells + region)

    for kind, (count, cells) in shapes.items():
        phrases.append((count_phrase(count, kind), cells))
    return phrases


def describe_block_structures(scene, id_to_char, block_chars, char_names):
    """
        Runs the same shape pass over the other block types, one material at a time,
        so a stack of bricks reads as a brick wall instead of a blob of bricks.
        Returns the (phrase, cells) pairs and the chars that produced a shape.
    """
    height = len(scene)
    width = len(scene[0]) if height else 0
    phrases = []
    named = set()

    for char in sorted(block_chars):
        name = char_names.get(char)
        if name is None:
            continue
        shapes = {}
        for region in terrain_regions(scene, id_to_char, {char}):
            kind = classify_terrain_region(region, height, width)
            if kind is None:
                continue
            noun = f"{name.lower()} {BLOCK_SHAPE_NOUNS[kind]}"
            count, cells = shapes.get(noun, (0, []))
            shapes[noun] = (count + 1, cells + region)
            named.add(char)
        for noun, (count, cells) in shapes.items():
            phrases.append((count_phrase(count, noun), cells))
    return phrases, named


def assign_caption(scene, id_to_char, char_names, ground_chars=None,
                   meta_phrases=None, debug=False, return_details=False,
                   block_chars=None):
    """
        Assigns a caption to a level scene based on its contents: the metadata, the
        ground summary, a count of each tile type, and a note for anything that piles
        up into a blob. Multi-tile objects now should count as one.
        Returns (caption, details) when return_details is True, where details maps
        each phrase to the (row, col) positions that produced it.
    """
    ground_chars = ground_chars or set()
    details = {} if return_details else None
    phrases = []

    def add_to_caption(phrase, contributing_blocks):
        if phrase:
            phrases.append(phrase)
            if return_details and details is not None:
                # The caption box splits on periods, so the keys keep theirs.
                details[f"{phrase}."] = contributing_blocks

    for phrase in meta_phrases or []:
        add_to_caption(phrase, [])      # metadata describes no tiles

    # Ground is covered by the floor phrase, so leave it out of the tile counts.
    cells = {}
    ground_cells = []
    for r, row in enumerate(scene):
        for c, tile_id in enumerate(row):
            char = id_to_char.get(tile_id)
            if char in ground_chars:
                ground_cells.append((r, c))
            elif char in char_names:
                cells.setdefault(char, []).append((r, c))

    add_to_caption(describe_ground(scene, id_to_char, ground_chars), ground_cells)

    for phrase, region in describe_terrain(scene, id_to_char, ground_chars):
        add_to_caption(phrase, region)

    shaped_chars = set()
    if block_chars:
        block_phrases, shaped_chars = describe_block_structures(
            scene, id_to_char, block_chars, char_names)
        for phrase, region in block_phrases:
            add_to_caption(phrase, region)

    blobs = largest_blobs(scene, id_to_char)
    object_counts = count_objects(scene, id_to_char)
    for char, char_cells in cells.items():
        name = char_names[char]
        count = object_counts.get(char, len(char_cells))
        add_to_caption(count_phrase(count, name), char_cells)
        # A pile of coins is a blob, but a row of bridges is just several bridges.
        # Blocks already named as a shape do not need one.
        if (char not in object_counts and char not in shaped_chars
                and len(blobs.get(char, ())) >= BLOB_THRESHOLD):
            add_to_caption(f"a blob of {pluralize(name)}".capitalize(), blobs[char])

    caption = " ".join(f"{p}." for p in phrases)
    return (caption, details) if return_details else caption


def generate_captions(dataset_path, tileset_path, output_path,
                      caption_mode="legacy", caption_key="deterministic_captions"):
    """Write a deterministic caption for every scene.

    "legacy" stores it in the "caption" field; "keyed" stores it as a one-element list under
    caption_key. Either way every other input attribute is copied through.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_char = build_id_to_char(tileset_path)
    char_names = get_char_names(tileset_path)
    _, _, ground_chars = get_tile_categories(tileset_path)
    block_chars = get_block_chars(tileset_path)

    captioned = []
    for item in dataset:
        is_dict = isinstance(item, dict)
        scene = item["scene"] if is_dict else item
        meta_phrases = metadata_phrases(item) if is_dict else []
        caption = assign_caption(scene, id_to_char, char_names, ground_chars,
                                 meta_phrases, block_chars=block_chars)
        entry = dict(item) if is_dict else {}  # copy all input attributes so metadata/other sources carry through
        entry["scene"] = scene
        if caption_mode == "keyed":
            entry[caption_key] = [caption]
        else:
            entry["caption"] = caption
        captioned.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captioned, f, indent=2, ensure_ascii=False)

    dest = f'"{caption_key}" list' if caption_mode == "keyed" else '"caption" field'
    print(f"Captioned {len(captioned)} scenes into the {dest} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Captions for MM2 ASCII datasets: a ground/floor summary plus per-tile counts and blob callouts.")
    parser.add_argument("--dataset", required=True, help="Input dataset JSON.")
    parser.add_argument("--tileset", required=True, help="Tileset JSON (e.g. mm2_tileset_we.json); names are read from its tile tags.")
    parser.add_argument("--output", required=True, help="Output captioned JSON.")
    parser.add_argument(
        "--caption-mode",
        choices=["legacy", "keyed"],
        default="legacy",
        help=(
            "Output schema. 'legacy' (default) writes the single 'caption' field. 'keyed' "
            "writes the caption as a one-element list under --caption-key, so a scene can carry "
            "captions from several sources at once. Both modes copy all other input attributes "
            "(metadata and captions from other sources) to the output."
        ),
    )
    parser.add_argument(
        "--caption-key",
        default="deterministic_captions",
        help="Key to store the caption list under when --caption-mode keyed. Default: deterministic_captions",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        print(f"Error: dataset not found: {args.dataset}")
        sys.exit(1)
    if not os.path.isfile(args.tileset):
        print(f"Error: tileset not found: {args.tileset}")
        sys.exit(1)

    generate_captions(args.dataset, args.tileset, args.output,
                      caption_mode=args.caption_mode, caption_key=args.caption_key)
