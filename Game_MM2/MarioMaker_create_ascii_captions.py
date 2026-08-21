import json
import os
import sys
import argparse
from collections import Counter

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
    """Largest contiguous same-tile region for each tile char, found by flood
    fill over 4-connected neighbours."""
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
            size = 0
            while stack:
                y, x = stack.pop()
                if (y, x) in visited or not (0 <= y < height and 0 <= x < width):
                    continue
                if id_to_char.get(scene[y][x]) != char:
                    continue
                visited.add((y, x))
                size += 1
                stack += [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
            if size > biggest.get(char, 0):
                biggest[char] = size
    return biggest


def describe_ground(scene, id_to_char, ground_chars):
    if not ground_chars:
        return None
    if not any(id_to_char.get(t) in ground_chars for row in scene for t in row):
        return None

    bottom = [id_to_char.get(t) in ground_chars for t in scene[-1]]
    if all(bottom):
        return "Full ground floor"
    if not any(bottom):
        return "Scattered ground"

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


def assign_caption(scene, id_to_char, char_names, ground_chars=None,
                   meta_phrases=None):
    # Metadata, the ground summary, then a count of every tile type present, and
    # a note for any type that piles up into a blob, e.g. "SMB1 style. Full
    # ground floor. Two goombas. A few coins. A blob of coins."
    ground_chars = ground_chars or set()
    phrases = list(meta_phrases) if meta_phrases else []

    ground_phrase = describe_ground(scene, id_to_char, ground_chars)
    if ground_phrase:
        phrases.append(ground_phrase)

    # Ground is covered by the floor phrase, so leave it out of the tile counts.
    counts = Counter()
    for row in scene:
        for tile_id in row:
            char = id_to_char.get(tile_id)
            if char and char not in ground_chars and char in char_names:
                counts[char] += 1

    blobs = largest_blobs(scene, id_to_char)
    for char, count in counts.items():
        name = char_names[char]
        phrases.append(count_phrase(count, name))
        if blobs.get(char, 0) >= BLOB_THRESHOLD:
            phrases.append(f"a blob of {pluralize(name)}".capitalize())

    return " ".join(f"{p}." for p in phrases)


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

    captioned = []
    for item in dataset:
        is_dict = isinstance(item, dict)
        scene = item["scene"] if is_dict else item
        meta_phrases = metadata_phrases(item) if is_dict else []
        caption = assign_caption(scene, id_to_char, char_names, ground_chars,
                                 meta_phrases)
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
