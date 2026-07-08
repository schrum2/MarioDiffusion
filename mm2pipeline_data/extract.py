"""Extract properly-formatted .bcd level files from the
TheGreatRambler/mm2_level HuggingFace dataset for use with Toost.

Usage
-----
    # Stream a small sample (recommended for testing):
    python -m mm2pipeline_data.extract --output_folder ./bcd_levels --limit 100

    # Extract specific data_ids:
    python -m mm2pipeline_data.extract --ids 3000004 3000007

    # Filter on server-side metadata:
    python -m mm2pipeline_data.extract --tag Speedrun --limit 50
    python -m mm2pipeline_data.extract --likes 1000 --dislikes 300 --exclude-tag Art

    # Extract everything (streaming, ~100 GB):
    python -m mm2pipeline_data.extract --output_folder ./bcd_levels

Besides the .bcd files, a `level_metadata.json` index (stem -> difficulty/tags)
is written to the output folder; mm2pipeline_data.toost folds it into the exported
level JSONs so the metadata survives the rest of the pipeline.

Requirements:
    pip install datasets pycryptodome
"""

import argparse
import json
from pathlib import Path

from .bcd import (
    PAYLOAD_SIZE,
    GAMESTYLE_SM3DW,
    build_bcd,
    decompress_level_data,
    get_gamestyle_raw,
    level_contains_skip_object,
    subworld_has_items,
)

# ---------------------------------------------------------------------------
# Course tags. Server-side metadata (dataset tag1/tag2 columns), not in the
# .bcd; 0 means no tag.
# ---------------------------------------------------------------------------
TAGS = {
    0: "None",
    1: "Standard",
    2: "Puzzle solving",
    3: "Speedrun",
    4: "Autoscroll",
    5: "Auto mario",
    6: "Short and sweet",
    7: "Multiplayer versus",
    8: "Themed",
    9: "Music",
    10: "Art",
    11: "Technical",
    12: "Shooter",
    13: "Boss battle",
    14: "Single player",
    15: "Link",  # probably not in v1.0.0, but listed so unknown values don't slip through
}


def level_tags(row) -> list:
    """Resolve a level's tag1/tag2 columns into a de-duplicated list of tag
    names. Empty slots are 0 ("None"); a level can have zero, one, or two tags."""
    names = []
    for key in ("tag1", "tag2"):
        value = row.get(key)
        if value in (None, 0):
            continue
        name = TAGS.get(value, f"unknown_{value}")
        if name not in names:
            names.append(name)
    return names


def resolve_tag_filter(tokens):
    """Turn the --tag CLI tokens into a set of lowercased tag names to match
    against. Each token may be a tag name (case-insensitive, e.g. "speedrun")
    or its numeric id (e.g. "3"). Raises SystemExit on an unknown tag."""
    name_by_lower = {name.lower(): name for value, name in TAGS.items() if value != 0}
    resolved = set()
    for token in tokens:
        key = token.strip().lower()
        if key.isdigit() and int(key) in TAGS and int(key) != 0:
            resolved.add(TAGS[int(key)].lower())
        elif key in name_by_lower:
            resolved.add(key)
        else:
            valid = ", ".join(TAGS[v] for v in sorted(TAGS) if v != 0)
            raise SystemExit(f"Unknown tag '{token}'. Valid tags: {valid}")
    return resolved


# ---------------------------------------------------------------------------
# Difficulty
# ---------------------------------------------------------------------------
# Difficulty is another server-side column (0-indexed) we can filter on.
DIFFICULTY = {
    0: "Easy",         # Rest of the levels with a clear rate > 30%
    1: "Normal",       # Levels with a clear rate < 30%
    2: "Expert",       # Levels with a clear rate < 10%
    3: "Super expert", # Levels with a clear rate < 2%
}


def resolve_difficulty_filter(tokens):
    """Turn the --difficulty CLI tokens into a set of difficulty ids to match
    against. Each token may be a difficulty name (case-insensitive, e.g.
    "normal") or its numeric id (e.g. "1"). Raises SystemExit on a bad value."""
    id_by_lower = {name.lower(): value for value, name in DIFFICULTY.items()}
    resolved = set()
    for token in tokens:
        key = token.strip().lower()
        if key.isdigit() and int(key) in DIFFICULTY:
            resolved.add(int(key))
        elif key in id_by_lower:
            resolved.add(id_by_lower[key])
        else:
            valid = ", ".join(f"{v}={DIFFICULTY[v]}" for v in sorted(DIFFICULTY))
            raise SystemExit(f"Unknown difficulty '{token}'. Valid difficulties: {valid}")
    return resolved


# ---------------------------------------------------------------------------
# Dataset extraction
# ---------------------------------------------------------------------------

def extract_levels(
    output_dir: str,
    limit=None,
    streaming: bool = True,
    data_id_filter=None,
    name_filter=None,
    name_count=None,
    tag_filter=None,
    tag_match_all=False,
    exclude_tag_filter=None,
    difficulty_filter=None,
    min_likes=None,
    max_dislikes=None,
    skip_3dworld: bool = False,
    skip_items: bool = False,
    skip_subworld_items: bool = False,
):
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset (streaming={streaming}) …")
    ds = load_dataset("TheGreatRambler/mm2_level", streaming=streaming, split="train")

    saved = skipped = errors = 0
    skipped_3dw = 0
    skipped_items = 0
    skipped_subworld = 0
    skipped_boos = 0
    skipped_likes = 0
    skipped_dislikes = 0
    skipped_excluded_tag = 0
    name_saved = 0

    # stem -> {difficulty, tags}, written to level_metadata.json; these are
    # server-side columns Toost can't emit, so they're carried forward here.
    metadata_index = {}

    for row in ds:
        data_id = row["data_id"]

        if data_id_filter is not None and data_id not in data_id_filter:
            continue

        if name_filter is not None:
            level_name = str(row.get("name", ""))
            if name_filter.lower() not in level_name.lower():
                continue

        # Resolve once so we can filter on tags and reuse them when saving.
        these_tags = level_tags(row)
        if tag_filter is not None:
            level_tag_set = {t.lower() for t in these_tags}
            if tag_match_all:
                if not tag_filter.issubset(level_tag_set):
                    continue
            elif not tag_filter & level_tag_set:
                continue

        # --exclude-tag: drop levels carrying any excluded tag.
        if exclude_tag_filter is not None:
            if exclude_tag_filter & {t.lower() for t in these_tags}:
                skipped_excluded_tag += 1
                continue

        if difficulty_filter is not None and row.get("difficulty") not in difficulty_filter:
            continue

        # Popularity gates (a missing count reads as 0).
        if min_likes is not None and (row.get("likes") or 0) < min_likes:
            skipped_likes += 1
            continue
        if max_dislikes is not None and (row.get("boos") or 0) > max_dislikes:
            skipped_dislikes += 1
            continue

        # Skip levels with more boos than likes.
        if (row.get("boos") or 0) > (row.get("likes") or 0):
            skipped_boos += 1
            continue

        raw = row["level_data"]
        if raw is None:
            skipped += 1
            continue

        try:
            plaintext = decompress_level_data(raw)
        except ValueError as e:
            print(f"  [WARN] data_id={data_id}: decompress error: {e}")
            errors += 1
            continue

        if len(plaintext) != PAYLOAD_SIZE:
            print(f"  [WARN] data_id={data_id}: unexpected size {len(plaintext)}, skipping")
            errors += 1
            continue

        if skip_3dworld and get_gamestyle_raw(plaintext) == GAMESTYLE_SM3DW:
            print(f"  [SKIP] data_id={data_id}: skipped because level is 3D World")
            skipped_3dw += 1
            continue

        if skip_items and level_contains_skip_object(plaintext):
            print(f"  [SKIP] data_id={data_id}: skipped because level contains a banned item")
            skipped_items += 1
            continue

        if skip_subworld_items and subworld_has_items(plaintext):
            print(f"  [SKIP] data_id={data_id}: skipped because subworld has items")
            skipped_subworld += 1
            continue

        try:
            bcd = build_bcd(plaintext)
        except Exception as e:
            print(f"  [WARN] data_id={data_id}: encrypt error: {e}")
            errors += 1
            continue

        if name_filter is not None:
            name_saved += 1
            filename = f"{data_id}_{name_saved}.bcd"
        else:
            filename = f"{data_id}.bcd"

        (out / filename).write_bytes(bcd)
        metadata_index[Path(filename).stem] = {
            "difficulty": DIFFICULTY.get(row.get("difficulty"), "Unknown"),
            "tags": these_tags,
        }
        saved += 1

        if name_filter is not None and name_count is not None:
            if name_saved >= name_count:
                break

        if saved % 500 == 0 or saved == 1:
            print(f"  Saved {saved} levels …  (last: {data_id}.bcd)")

        if limit is not None and saved >= limit:
            break

    (out / "level_metadata.json").write_text(
        json.dumps(metadata_index, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"\nDone. Saved: {saved}  |  Skipped (null): {skipped}  |  "
        f"Skipped (3D World): {skipped_3dw}  |  Skipped (banned item): {skipped_items}  |  "
        f"Skipped (subworld has items): {skipped_subworld}  |  "
        f"Skipped (more boos than likes): {skipped_boos}  |  "
        f"Skipped (too few likes): {skipped_likes}  |  "
        f"Skipped (too many dislikes): {skipped_dislikes}  |  "
        f"Skipped (excluded tag): {skipped_excluded_tag}  |  "
        f"Errors: {errors}"
    )
    print(f"Output dir: {out.resolve()}")
    print(f"Wrote metadata for {len(metadata_index)} level(s) -> {(out / 'level_metadata.json').resolve()}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Extract properly-formatted .bcd level files from the "
            "TheGreatRambler/mm2_level HuggingFace dataset for use with Toost."
        )
    )
    p.add_argument("--output_folder", "-o", default="./bcd_levels")
    p.add_argument("--limit", "-n", type=int, default=None,
                   help="Max levels to extract (default: all)")
    p.add_argument("--no_stream", action="store_true",
                   help="Download full dataset first (~100 GB)")
    p.add_argument("--ids", nargs="+", type=int, default=None,
                   metavar="DATA_ID", help="Only extract these data_ids")
    p.add_argument("--name", type=str, default=None, help="Extract levels whose name contains this text")
    p.add_argument("--name_count", type=int, default=None, help="Number of matching levels to extract")
    p.add_argument("--tag", nargs="+", default=None, metavar="TAG",
                   help="Only extract levels carrying at least one of these tags "
                        "(name or id, e.g. --tag Speedrun \"Short and sweet\"). "
                        "Tags: " + ", ".join(TAGS[v] for v in sorted(TAGS) if v != 0))
    p.add_argument("--all-tags", action="store_true",
                   help="Require every --tag to be present, not just one. A level "
                        "can have at most 2 tags, so passing more than 2 is an error.")
    p.add_argument("--exclude-tag", nargs="+", default=None, metavar="TAG",
                   help="Skip levels carrying any of these tags (name or id, e.g. "
                        "--exclude-tag Art Music). Applied after --tag.")
    p.add_argument("--difficulty", nargs="+", default=None, metavar="DIFFICULTY",
                   help="Only extract levels at one of these difficulties "
                        "(name or id, e.g. --difficulty Easy Normal). "
                        "Difficulties: " + ", ".join(f"{v}={DIFFICULTY[v]}" for v in sorted(DIFFICULTY)))
    p.add_argument("--likes", type=int, default=None, metavar="N",
                   help="Only extract levels with at least N likes (e.g. --likes 1000)")
    p.add_argument("--dislikes", type=int, default=None, metavar="N",
                   help="Only extract levels with at most N dislikes/boos (e.g. --dislikes 300)")
    p.add_argument("--skip_3dworld", action="store_true",
                   help="Skip levels whose gamestyle is Super Mario 3D World, needed for .swe conversion.")
    p.add_argument("--skip_items", action="store_true",
                   help="Skip levels containing any object listed in SKIP_ITEM_NAMES, needed for .swe conversion.")
    p.add_argument("--skip_subworld_items", action="store_true",
                   help="Skip levels whose subworld contains any items, needed for .swe conversion.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    tag_filter = resolve_tag_filter(args.tag) if args.tag else None
    if args.all_tags:
        if not tag_filter:
            raise SystemExit("--all-tags only makes sense together with --tag.")
        if len(tag_filter) > 2:
            raise SystemExit("--all-tags takes at most 2 tags (a level can't have more than 2).")

    exclude_tag_filter = resolve_tag_filter(args.exclude_tag) if args.exclude_tag else None
    # Asking to both keep and drop the same tag can never match anything, so
    # flag it up front instead of silently returning zero levels.
    if tag_filter and exclude_tag_filter and (tag_filter & exclude_tag_filter):
        clash = ", ".join(sorted(tag_filter & exclude_tag_filter))
        raise SystemExit(f"--tag and --exclude-tag can't share a tag: {clash}")

    extract_levels(
        output_dir=args.output_folder,
        limit=args.limit,
        streaming=not args.no_stream,
        data_id_filter=set(args.ids) if args.ids else None,
        name_filter=args.name,
        name_count=args.name_count,
        tag_filter=tag_filter,
        tag_match_all=args.all_tags,
        exclude_tag_filter=exclude_tag_filter,
        difficulty_filter=resolve_difficulty_filter(args.difficulty) if args.difficulty else None,
        min_likes=args.likes,
        max_dislikes=args.dislikes,
        skip_3dworld=args.skip_3dworld,
        skip_items=args.skip_items,
        skip_subworld_items=args.skip_subworld_items,
    )


if __name__ == "__main__":
    main()
