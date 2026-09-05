"""Deterministically score LLM captions against the tile content of each scene.

This is a lightweight grounding metric, not a truth oracle.  It rewards captions that
mention semantic categories present in a scene (for example water, enemies, hazards,
and power-ups), gives extra evidence for tile-specific vocabulary (for example
"Kamadoma"), and penalizes recognized concepts that are absent from the scene.

Example:
    python evaluate_llm_caption_grounding.py --input captions.json \
        --game MM-Full --caption-key gemma4:12b_captions --output scored.json
"""

import argparse
from collections import Counter
import json
import re
from pathlib import Path

from captions.util import extract_tileset
from util.descriptive_tilesets import GAMES


# These are intentionally conservative.  The tile description supplies the more specific
# vocabulary; these words let a caption say "enemies" instead of naming every enemy.
CATEGORY_TERMS = {
    "enemy": {"enemy", "enemies", "foe", "foes"},
    "hazard": {"hazard", "hazards", "danger", "dangers", "obstacle", "obstacles"},
    "powerup": {"powerup", "powerups", "power", "powers", "collectible", "collectibles", "item", "items"},
    "platform": {"platform", "platforms", "lift", "lifts"},
    "block": {"block", "blocks", "brick", "bricks"},
    "ladder": {"ladder", "ladders"},
    "door": {"door", "doors"},
    "goal": {"goal", "goals", "exit", "exits", "flag", "flagpole"},
    "water": {"water"},
    "lava": {"lava"},
    "spring": {"spring", "springs"},
    "coin": {"coin", "coins"},
}

IGNORED_DESCRIPTION_WORDS = {
    "a", "an", "and", "as", "but", "can", "collectible", "damaging", "deadly",
    "enemy", "fades", "from", "ground", "in", "like", "of", "out", "passable",
    "power", "represents", "solid", "the", "this", "to", "type", "up", "with",
    "moving", "ranged", "stationary", "horizontal", "vertical", "large", "small",
    "appearing", "depending", "final", "game", "interactive", "looks", "man", "mega",
    "one", "player", "regular", "reappearing", "secret", "shortly", "specific", "starting",
    "style", "temporary", "transparent", "way", "when",
}


def normalize_word(word: str) -> str:
    """Normalize a word enough for common singular/plural variants."""
    word = word.lower().replace("-", "")
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [normalize_word(token) for token in re.findall(r"[a-z0-9]+", text.lower())]


def phrase_present(tokens: list[str], phrase: str) -> bool:
    wanted = tokenize(phrase)
    if not wanted:
        return False
    return any(tokens[index:index + len(wanted)] == wanted
               for index in range(len(tokens) - len(wanted) + 1))


def tile_terms(description: str) -> set[str]:
    """Extract useful distinctive words from a descriptive tileset entry."""
    words = re.findall(r"[a-z0-9]+", description.lower())
    return {word for word in words if word not in IGNORED_DESCRIPTION_WORDS and len(word) > 2}


def category_for_tile(description: str, tags: set[str]) -> set[str]:
    lowered = description.lower()
    categories = set()
    if "enemy" in tags:
        categories.add("enemy")
    if "hazard" in tags:
        categories.add("hazard")
    if "powerup" in tags or "power-up" in tags or "collectable" in tags:
        categories.add("powerup")
    if "platform" in tags or "moving" in tags and "platform" in lowered:
        categories.add("platform")
    if "block" in lowered or "brick" in lowered:
        categories.add("block")
    for category in ("ladder", "door", "water", "lava", "spring", "coin"):
        if category in lowered:
            categories.add(category)
    if "goal" in lowered or "exit" in lowered or "flagpole" in lowered:
        categories.add("goal")
    return categories


def build_vocabulary(game: str, id_to_char: dict[int, str], tile_descriptors: dict) -> dict:
    """Build category and tile-specific concepts from the registered game tileset."""
    descriptions = GAMES[game]["tiles"]["tiles"]
    vocabulary = {category: set(terms) for category, terms in CATEGORY_TERMS.items()}
    tile_concepts = {}
    for char, description in descriptions.items():
        if char not in id_to_char.values():
            continue
        tags = set(tile_descriptors.get(char, set()))
        tile_concepts[char] = {
            "description": description,
            "terms": tile_terms(description),
            "categories": category_for_tile(description, tags),
        }
    return {"categories": vocabulary, "tiles": tile_concepts}


def scene_characters(scene: list[list[int]], id_to_char: dict[int, str]) -> Counter:
    return Counter(id_to_char[tile] for row in scene for tile in row if tile in id_to_char)


def score_caption(caption: str, scene: list[list[int]], id_to_char: dict[int, str], vocabulary: dict) -> dict:
    """Return interpretable coverage, precision, and grounding scores in [0, 1]."""
    tokens = tokenize(caption)
    present = scene_characters(scene, id_to_char)
    present_categories = set()
    present_tiles = set(present)
    for char in present_tiles:
        present_categories.update(vocabulary["tiles"].get(char, {}).get("categories", set()))

    # Evaluate category alternatives explicitly while preserving a stable, human-readable result.
    required = sorted(category for category in present_categories
                      if any(phrase_present(tokens, term) for term in vocabulary["categories"][category]))
    all_categories = sorted(vocabulary["categories"])
    mentioned_categories = sorted(category for category in all_categories
                                  if any(phrase_present(tokens, term) for term in vocabulary["categories"][category]))
    unsupported_categories = sorted(set(mentioned_categories) - present_categories)

    specific_matches = []
    for char, info in vocabulary["tiles"].items():
        if any(phrase_present(tokens, term) for term in info["terms"]):
            specific_matches.append({"char": char, "description": info["description"]})
    supported_specific = [item for item in specific_matches if item["char"] in present_tiles]
    unsupported_specific = [item for item in specific_matches if item["char"] not in present_tiles]

    # Category coverage is the main score. Specific terms are a bonus signal and do not make
    # omission of every exact enemy type look like a failure when "enemies" is accurate.
    coverage = len(required) / len(present_categories) if present_categories else 1.0
    mentioned_count = len(mentioned_categories) + len(specific_matches)
    unsupported_count = len(unsupported_categories) + len(unsupported_specific)
    precision = ((mentioned_count - unsupported_count) / mentioned_count
                 if mentioned_count else 1.0)
    overall = (2 * coverage * precision / (coverage + precision)
               if coverage + precision else 0.0)
    return {
        "coverage": round(coverage, 6),
        "precision": round(max(0.0, precision), 6),
        "overall": round(overall, 6),
        "present_categories": sorted(present_categories),
        "mentioned_categories": mentioned_categories,
        "unsupported_categories": unsupported_categories,
        "supported_specific_tiles": supported_specific,
        "unsupported_specific_tiles": unsupported_specific,
        "scene_tile_counts": dict(present),
    }


def load_entries(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of scene entries")
    return data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON output from llm_ascii_to_caption.py")
    parser.add_argument("--game", required=True, choices=sorted(GAMES), help="Registered game/tileset")
    parser.add_argument("--caption-key", required=True, help="Entry key containing a caption string or list")
    parser.add_argument("--output", help="Optional path for annotated per-caption JSON")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most this many scene entries")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    game_config = GAMES[args.game]
    _, id_to_char, _, tile_descriptors = extract_tileset(game_config["tileset"])
    vocabulary = build_vocabulary(args.game, id_to_char, tile_descriptors)
    entries = load_entries(args.input)[:args.limit]

    scored_entries = []
    scores = []
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "scene" not in entry or args.caption_key not in entry:
            continue
        captions = entry[args.caption_key]
        captions = captions if isinstance(captions, list) else [captions]
        caption_scores = []
        for caption in captions:
            if not isinstance(caption, str):
                continue
            result = score_caption(caption, entry["scene"], id_to_char, vocabulary)
            result["caption"] = caption
            caption_scores.append(result)
            scores.append(result["overall"])
        if caption_scores:
            scored_entries.append({"entry_index": entry_index, "scores": caption_scores})

    summary = {
        "game": args.game,
        "caption_key": args.caption_key,
        "scene_entries": len(scored_entries),
        "caption_count": len(scores),
        "average_coverage": sum(item["coverage"] for entry in scored_entries for item in entry["scores"]) / len(scores) if scores else None,
        "average_precision": sum(item["precision"] for entry in scored_entries for item in entry["scores"]) / len(scores) if scores else None,
        "average_overall": sum(scores) / len(scores) if scores else None,
        "entries": scored_entries,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote caption grounding scores to {args.output}")
    else:
        print(json.dumps({key: value for key, value in summary.items() if key != "entries"}, indent=2))
    return summary


if __name__ == "__main__":
    main()