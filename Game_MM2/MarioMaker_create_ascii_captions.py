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
GROUND_WIDTH_FRACTION = 0.3

# Surface heights within this much of each other read as level ground.
SURFACE_FLAT_TOLERANCE = 2

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

# Below this much solid along a row there is no ceiling worth describing.
CEILING_MIN_COVERAGE = 0.5

# A window can cut in just under the ceiling, so check a few rows down for it.
CEILING_SCAN_ROWS = 3

# A line or column of loose tiles needs this many before it reads as one.
ARRANGEMENT_MIN_RUN = 3

# Loose tiles with no shape need this many before they count as a clump.
CLUMP_MIN = 5

# This many separate little groups of one tile type read as scattered.
SCATTER_GROUPS = 4

# Tags that keep a tile out of the block set: it moves, hurts, or is picked up.
BLOCK_EXCLUDE_TAGS = {"damaging", "hazard", "enemy", "moving", "warp", "pipe",
                      "shooter", "collectable", "power-up"}

# Shape names for structures built out of blocks instead of ground.
BLOCK_SHAPE_NOUNS = {
    "ground": "floor",
    "platform": "platform",
    "ledge": "ledge",
    "tower": "tower",
    "column": "column",
    "ground block": "wall",
    "staircase": "staircase",
    "hollow structure": "structure",
    "mound": "cluster",
}

# What a group of loose tiles is called once its arrangement is known.
ARRANGEMENT_NOUNS = {"row": "line", "column": "column", "block": "cluster"}


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
    """Map each tile char to the name toost decodes it as, so a caption calls
    something the same thing the level data does."""
    from mm2pipeline_data.tiles import CHAR_TO_NAME

    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    char_names = {}
    for char, tags in data["tiles"].items():
        if not tags or any(t in EMPTY_TAGS for t in tags):
            continue
        name = CHAR_TO_NAME.get(char)
        if name is None:
            # A glyph toost has no name for. The last non-property tag names it,
            # so a boss (Bowser is [..., "boss", "bowser"]) isn't named "boss".
            name_tags = [t for t in tags if t not in PROPERTY_TAGS]
            name = (name_tags[-1] if name_tags else tags[-1]).title()
        char_names[char] = name
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
    """Tile chars that build structures the way ground does, minus the ground
    itself. Pipes and bridges stay out; a footprint already counts those once."""
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


def get_loose_chars(tileset_path):
    """Tile chars that sit loose in a scene rather than building it: enemies,
    pickups and hazards. Anything with its own footprint is already one object."""
    from util.mm2_metrics import FEATURE_POLICIES

    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    block_chars = get_block_chars(tileset_path)
    chars = set()
    for char, tags in data["tiles"].items():
        if char in FEATURE_POLICIES or char in block_chars:
            continue
        if "ground" in tags or any(t in EMPTY_TAGS for t in tags):
            continue
        chars.add(char)
    return chars


def get_solid_chars(tileset_path):
    """Tile chars tagged solid, which is what a region can rest on or be buried in."""
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {char for char, tags in data["tiles"].items() if "solid" in tags}


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


def describe_ceiling(scene, id_to_char, solid_chars):
    if not scene or not scene[0]:
        return None

    for i, row in enumerate(scene[:CEILING_SCAN_ROWS]):
        # A few stray blocks overhead are not a ceiling, and neither is the top of
        # a mass of ground with no room under it.
        top = [id_to_char.get(t) in solid_chars for t in row]
        if sum(top) < CEILING_MIN_COVERAGE * len(top):
            continue
        below = scene[i + 1] if i + 1 < len(scene) else []
        if sum(id_to_char.get(t) in solid_chars for t in below) >= CEILING_MIN_COVERAGE * len(top):
            continue
        if all(top):
            return "Full ceiling"

        gaps = 0
        in_gap = False
        for is_solid in top:
            if not is_solid:
                gaps += not in_gap
                in_gap = True
            else:
                in_gap = False
        return f"Ceiling with {describe_quantity(gaps)} gap" + ("s" if gaps > 1 else "")
    return None


def classify_arrangement(cells):
    """Names how a group of loose tiles is laid out: "row", "column", "block",
    "scattered", or None when it is too small or sparse to mention."""
    if len(cells) < 2:
        return None
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    span_w = max(cols) - min(cols) + 1
    span_h = max(rows) - min(rows) + 1

    if span_h == 1 and span_w >= ARRANGEMENT_MIN_RUN:
        return "row"
    if span_w == 1 and span_h >= ARRANGEMENT_MIN_RUN:
        return "column"
    if span_w >= 2 and span_h >= 2 and len(cells) >= SOLID_FILL_FRACTION * span_w * span_h:
        return "block"
    if len(cells) >= CLUMP_MIN:
        return "clump"
    return None


def describe_arrangements(scene, id_to_char, loose_chars, char_names):
    """Runs over the loose tiles one type at a time, so a row of coins becomes a
    line of coins. Returns (phrase, cells) pairs and the chars that got one."""
    phrases = []
    named = set()

    for char in sorted(loose_chars):
        name = char_names.get(char)
        if name is None:
            continue
        shapes = {}
        loose = []
        groups = 0
        for region in terrain_regions(scene, id_to_char, {char}):
            kind = classify_arrangement(region)
            if kind is None:
                loose.extend(region)
                groups += 1
                continue
            if kind == "clump":
                phrases.append((f"A clump of {pluralize(name.lower())}", region))
                named.add(char)
                continue
            count, cells = shapes.get(kind, (0, []))
            shapes[kind] = (count + 1, cells + region)
            named.add(char)
        # "Two lines of coins" rather than "two coins lines", since several tile
        # names are already plural.
        for kind, (count, cells) in shapes.items():
            head = ARRANGEMENT_NOUNS[kind]
            head = pluralize(head) if count > 1 else head
            phrase = f"{describe_quantity(count)} {head} of {pluralize(name.lower())}"
            phrases.append((phrase.capitalize(), cells))
        if groups >= SCATTER_GROUPS:
            phrases.append((f"Scattered {pluralize(name.lower())}", loose))
            named.add(char)
    return phrases, named


def terrain_regions(scene, id_to_char, terrain_chars):
    """Flood fills connected regions of terrain over 4-connected neighbours, every
    char in terrain_chars counting as one material. Largest region first."""
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
    """The highest cell in each column a region covers, left to right."""
    tops = {}
    for r, c in cells:
        if c not in tops or r < tops[c]:
            tops[c] = r
    return [tops[c] for c in sorted(tops)]


def describe_surface(profile):
    """Names a region's top edge: "flat", "rising to the right", "sloping down to
    the right" or "uneven". Rows count downward, so a smaller row is higher up."""
    if len(profile) < 2 or max(profile) - min(profile) <= SURFACE_FLAT_TOLERANCE:
        return "flat"
    if all(b <= a for a, b in zip(profile, profile[1:])):
        return "rising to the right"
    if all(b >= a for a, b in zip(profile, profile[1:])):
        return "sloping down to the right"
    return "uneven"


def solid_positions(scene, id_to_char, solid_chars):
    """Every solid position in the scene, used to see what a region is touching."""
    return {(r, c) for r, row in enumerate(scene) for c, tile_id in enumerate(row)
            if id_to_char.get(tile_id) in solid_chars}


def surface_is_clear(cells, solid):
    """True when nothing solid rests on a region's top edge, checked column by
    column so a brick course buried in a wall is not mistaken for a platform."""
    tops = {}
    for r, c in cells:
        if c not in tops or r < tops[c]:
            tops[c] = r
    return not any((r - 1, c) in solid for c, r in tops.items() if r > 0)


def base_is_clear(cells, solid, height):
    """True when nothing solid holds a region up, which separates a platform you
    can pass under from a ledge sitting on something."""
    bottoms = {}
    for r, c in cells:
        if c not in bottoms or r > bottoms[c]:
            bottoms[c] = r
    return not any((r + 1, c) in solid for c, r in bottoms.items() if r < height - 1)


def undercut_columns(cells, height):
    """Counts the columns with empty space directly below the region, which is
    what a cave roof or a tunnel looks like from above."""
    occupied = set(cells)
    undercut = set()
    for r, c in cells:
        if r + 1 < height and (r + 1, c) not in occupied:
            undercut.add(c)
    return len(undercut)


def classify_terrain_region(cells, height, width, solid=None):
    """
        Sorts a terrain region into a shape by its footprint and by what it touches,
        so a tall thin one standing on the floor is a tower rather than a column.
        Returns the shape name, or None when the region is too small or buried.
    """
    if len(cells) < TERRAIN_MIN_REGION:
        return None
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    span_w = max(cols) - min(cols) + 1
    span_h = max(rows) - min(rows) + 1
    on_floor = max(rows) == height - 1
    clear_above = solid is None or surface_is_clear(cells, solid)
    clear_below = solid is None or base_is_clear(cells, solid, height)

    if on_floor and span_w >= width * GROUND_WIDTH_FRACTION:
        return "ground"
    if span_h <= 2 and span_w >= 3:
        # Buried in something bigger, so the tile count already speaks for it.
        if not clear_above:
            return None
        return "platform" if clear_below else "ledge"
    if span_w <= 2 and span_h >= 3:
        # Standing on something is what makes it a tower rather than a hanging column.
        return "tower" if on_floor or not clear_below else "column"
    if len(cells) >= SOLID_FILL_FRACTION * span_w * span_h:
        return "ground block"
    # A top edge that climbs the whole way without turning back is a staircase,
    # as long as you could actually land on it.
    if span_w >= STAIRCASE_MIN_SPAN and span_h >= STAIRCASE_MIN_SPAN and clear_above:
        if describe_surface(surface_profile(cells)) in SLOPED_SURFACES:
            return "staircase"
    # Big and mostly empty inside: walls around rooms or a maze, not a clump.
    if span_w >= 6 and span_h >= 4:
        return "hollow structure"
    return "mound"


def describe_terrain(scene, id_to_char, terrain_chars, solid=None):
    """Describes the terrain above the bottom row, which the floor summary cannot
    reach. Returns (phrase, cells) pairs in the order they should be read."""
    height = len(scene)
    width = len(scene[0]) if height else 0
    phrases = []
    shapes = {}

    for region in terrain_regions(scene, id_to_char, terrain_chars):
        kind = classify_terrain_region(region, height, width, solid)
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


def describe_block_structures(scene, id_to_char, block_chars, char_names, solid=None):
    """Runs the shape pass over the other block types, one material at a time, so
    a stack of bricks reads as a brick wall. Same returns as describe_arrangements."""
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
            kind = classify_terrain_region(region, height, width, solid)
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
                   block_chars=None, solid_chars=None, loose_chars=None):
    """
        Assigns a caption to a level scene based on its contents: the metadata, the
        floor and ceiling, the shape of the terrain and whatever is built on it, how
        the loose tiles are arranged, and a count of each tile type.
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
    if solid_chars:
        add_to_caption(describe_ceiling(scene, id_to_char, solid_chars),
                       [(0, c) for c in range(len(scene[0]))] if scene else [])

    solid = solid_positions(scene, id_to_char, solid_chars) if solid_chars else None

    for phrase, region in describe_terrain(scene, id_to_char, ground_chars, solid):
        add_to_caption(phrase, region)

    if block_chars:
        block_phrases, _ = describe_block_structures(
            scene, id_to_char, block_chars, char_names, solid)
        for phrase, region in block_phrases:
            add_to_caption(phrase, region)

    if loose_chars:
        loose_phrases, _ = describe_arrangements(
            scene, id_to_char, loose_chars, char_names)
        for phrase, region in loose_phrases:
            add_to_caption(phrase, region)

    object_counts = count_objects(scene, id_to_char)
    for char, char_cells in cells.items():
        name = char_names[char]
        count = object_counts.get(char, len(char_cells))
        add_to_caption(count_phrase(count, name), char_cells)

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
    solid_chars = get_solid_chars(tileset_path)
    loose_chars = get_loose_chars(tileset_path)

    captioned = []
    for item in dataset:
        is_dict = isinstance(item, dict)
        scene = item["scene"] if is_dict else item
        meta_phrases = metadata_phrases(item) if is_dict else []
        caption = assign_caption(scene, id_to_char, char_names, ground_chars,
                                 meta_phrases, block_chars=block_chars,
                                 solid_chars=solid_chars, loose_chars=loose_chars)
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
