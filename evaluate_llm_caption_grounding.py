"""Deterministically score LLM captions against the tile content of each scene.

This is a lightweight grounding metric, not a truth oracle.  It rewards captions that
mention semantic categories present in a scene (for example water, enemies, hazards,
and power-ups), gives extra evidence for tile-specific vocabulary (for example
"Kamadoma"), and penalizes recognized concepts that are absent from the scene.

Example:
    python evaluate_llm_caption_grounding.py --input captions.json --game MM-Full --caption-key gemma4:12b_captions --output scored.json
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
    "hazard": {"hazard", "hazards", "danger", "dangers", "obstacle", "obstacles", "spikes", "spike", "spiked", "trap", "traps"},
    "powerup": {"powerup", "powerups", "power", "powers", "collectible", "collectibles", "item", "items", "pickup", "pickups"},
    "platform": {"platform", "platforms"},
    # "block": {"block", "blocks", "brick", "bricks"}, # These are just the general floor tiles. They are so common that specifically mentioning them is not useful.
    "ladder": {"ladder", "ladders"},
    "door": {"door", "doors", "doorway", "doorways", "gate", "gates"},
    "pipe": {"pipe", "pipes"},
    "water": {"water"},
    "lava": {"lava"},
    "spring": {"spring", "springs"},
    "coin": {"coin", "coins"},
}

# Context-sensitive category terms. These words should not be treated as standalone
# category claims when they modify a more specific noun in the caption.
CATEGORY_EXCLUSIONS = {
    "powerup": {
        "key": {"key door", "key doors"},
    },
}

SPECIFIC_TERM_EXCLUSIONS = {
    "key": {"key door", "key doors"},
    "mushroom": {"mushroom platform", "platform mushroom", "platform of mushroom"},
}

# These concepts are meaningful only as phrases. A lone "block" or "platform" is
# intentionally too broad, while "breakable block" and "moving platform" carry
# useful information about the scene.
COMBINED_CONCEPTS = {
    "breakable block": {"breakable block", "breakable brick", "brick block"},
    "transparent block": {"secret block", "transparent block"},
    "disappearing block": {"disappearing block", "reappearing block"},
    "moving lift": {"moving block", "moving platform", "moving lift", "lift"},
    "falling platform": {"falling platform"},
    "fake block": {"fake block"},
    "question block": {"question block"},
    "note block": {"note block"},
    "mushroom platform": {"mushroom platform", "platform mushroom", "platform of mushroom"},
    "semisolid platform": {"semisolid platform", "semi-solid platform"},
    "hidden block": {"hidden block"},
    "donut block": {"donut block"},
    "ice block": {"ice block", "slippery block"},
    "on off block": {"on/off block", "on off block"},
    "dotted line block": {"dotted-line block", "dotted line block"},
    "moving lift": {"moving lift"},
    "fading platform": {"fading platform", "fading platforms"},
    "life energy": {"life energy", "health energy"},
    "weapon energy": {"weapon energy"},
    "extra life": {"extra life", "1 up", "1-up"},
    "magnet beam": {"magnet beam"},
    "yashichi": {"yashichi"},
}

COMBINED_CATEGORY_CONCEPTS = {
    "fire hazard": {
        "fire hazard", "flame hazard", "fire hazards", "flame hazards", "burner hazard",
    },
}

POWERUP_COMPOUND_CONCEPTS = {
    "life energy", "weapon energy", "extra life", "magnet beam", "yashichi",
}

# Modifier+noun phrases provide bonus specificity but are never required for the
# generic category score. The matcher accepts hyphenated and spaced spellings.
SPECIFICITY_PHRASES = {
    "enemy": {
        "jumping enemy", "flying enemy", "ranged enemy", "vertical enemy",
        "horizontal enemy", "stationary enemy", "ground walking enemy",
        "moving enemy", "floating enemy",
    },
    "powerup": {
        "large powerup", "small powerup", "large power", "small power",
        "weapon powerup", "weapon power", "life powerup", "life power",
        "health powerup", "health power", "large pickup", "small pickup",
        "weapon pickup", "life pickup", "health pickup",
    },
}

UNSUPPORTED_SPECIFICITY_PENALTY = 0.25

IGNORED_DESCRIPTION_WORDS = {
    "a", "an", "and", "as", "but", "can", "collectible", "damaging", "deadly",
    "enemy", "fades", "from", "ground", "in", "like", "of", "out", "passable",
    "power", "represents", "solid", "the", "this", "to", "type", "up", "with",
    "moving", "ranged", "stationary", "horizontal", "vertical", "large", "small",
    "appearing", "depending", "final", "game", "interactive", "looks", "man", "mega",
    "one", "player", "regular", "reappearing", "secret", "shortly", "specific", "starting",
    "style", "temporary", "transparent", "way", "when", "that", "right", "left", "path", "track",
    "rail", "opens", "opened", "behaves", "barrier", "shooting", "pushes", "warps", "paired",
    "block", "blocks", "brick", "bricks",
    "energy", "life", "weapon", "extra",
    "fire",
    "drop", "drops", "gap", "gaps", "elevated", "early", "end", "middle", "sections",
    # Behaviour, position, appearance, and generic physical-property words are not
    # reliable evidence for a particular tile. For example, "floating" can describe
    # platforms or islands and must not imply the Watcher tile.
    "jumping", "jump", "flying", "floating", "exploding", "walking", "rising", "falling",
    "slows", "blows", "tackle", "extends", "shoots", "rides", "riding", "hidden", "fake",
    "breakable", "climbable", "damaging", "deadly", "solid", "passable", "transparent",
    "large", "small", "vertical", "horizontal", "temporary", "periodic", "rotating",
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
    tokens = [normalize_word(token) for token in re.findall(r"[a-z0-9]+", text.lower())]
    # Treat the common spaced spelling "power up" like the hyphenated and closed
    # spellings "power-up" and "powerup".
    merged = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index:index + 2] == ["power", "up"]:
            merged.append("powerup")
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return merged


def phrase_present(tokens: list[str], phrase: str) -> bool:
    wanted = tokenize(phrase)
    if not wanted:
        return False
    return any(tokens[index:index + len(wanted)] == wanted
               for index in range(len(tokens) - len(wanted) + 1))


def category_term_present(tokens: list[str], category: str, term: str) -> bool:
    """Match a category term unless it is used in a more specific excluded phrase."""
    if not phrase_present(tokens, term):
        return False
    exclusions = CATEGORY_EXCLUSIONS.get(category, {}).get(normalize_word(term), set())
    return not any(phrase_present(tokens, phrase) for phrase in exclusions)


def specific_term_present(tokens: list[str], term: str) -> bool:
    """Match a tile term unless it is being used inside an excluded compound phrase."""
    return phrase_present(tokens, term) and not any(
        phrase_present(tokens, phrase)
        for phrase in SPECIFIC_TERM_EXCLUSIONS.get(normalize_word(term), set())
    )


def tile_name_terms(description: str) -> set[str]:
    """Extract proper-name anchors, excluding generic prose from tile descriptions."""
    words = re.findall(r"[A-Z][A-Za-z0-9']*", description)
    return {
        word.lower() for word in words
        if normalize_word(word) not in IGNORED_DESCRIPTION_WORDS and len(word) > 2
    }


def scene_has_platform(scene: list[list[int]], id_to_char: dict[int, str],
                       tile_descriptors: dict, tile_concepts: dict) -> bool:
    """Detect explicit platform tiles or a Mario-style horizontal solid platform run."""
    present_chars = {
        id_to_char[tile]
        for row in scene
        for tile in row
        if tile in id_to_char
    }
    for char, tags in tile_descriptors.items():
        if char in present_chars and (
                "platform" in tags
                or "platform" in tile_concepts.get(char, {}).get("description", "").lower()):
            return True

    height = len(scene)
    width = len(scene[0]) if height else 0

    def is_open(row: int, col: int) -> bool:
        descriptors = tile_descriptors.get(id_to_char.get(scene[row][col]), set())
        return "solid" not in descriptors and "null" not in descriptors

    for row in range(max(0, height - 1)):
        col = 0
        while col < width:
            char = id_to_char.get(scene[row][col])
            if "solid" not in tile_descriptors.get(char, set()) or "pipe" in tile_descriptors.get(char, set()):
                col += 1
                continue
            start = col
            while col < width:
                current = id_to_char.get(scene[row][col])
                descriptors = tile_descriptors.get(current, set())
                if "solid" not in descriptors or "pipe" in descriptors:
                    break
                above_open = row > 0 and is_open(row - 1, col)
                below_open = row + 1 < height and is_open(row + 1, col)
                if not (above_open and below_open):
                    break
                col += 1
            if col - start >= 2:
                return True
            col = max(col + 1, start + 1)
    return False


def matching_phrases(description: str, tags: set[str]) -> set[str]:
    """Return phrase-level concepts that identify this tile without broad adjectives."""
    lowered = description.lower()
    phrases = set()
    for concept, alternatives in COMBINED_CONCEPTS.items():
        if any(phrase_present(tokenize(lowered), alternative) for alternative in alternatives):
            phrases.add(concept)
    if "disappearing" in lowered or "reappearing" in lowered:
        phrases.add("fading platform")
    categories = category_for_tile(description, tags)
    for category, alternatives in SPECIFICITY_PHRASES.items():
        if category not in categories:
            continue
        for phrase in alternatives:
            modifier, noun = phrase.rsplit(" ", 1)
            # The modifier and category noun need not be adjacent in the tileset
            # description, but they must be adjacent in the caption.
            description_tokens = tokenize(lowered)
            modifier_tokens = tokenize(modifier)
            if any(description_tokens[i:i + len(modifier_tokens)] == modifier_tokens
                   for i in range(len(description_tokens) - len(modifier_tokens) + 1)):
                phrases.add(phrase)
    return phrases


def category_for_tile(description: str, tags: set[str]) -> set[str]:
    lowered = description.lower()
    categories = set()
    if "enemy" in tags:
        categories.add("enemy")
    if "hazard" in tags:
        categories.add("hazard")
    is_platform = "platform" in tags or "platform" in lowered
    is_coin = "coin" in lowered
    if ("powerup" in tags or "power-up" in tags or "collectable" in tags) and not is_coin and not is_platform:
        categories.add("powerup")
    if is_platform or "moving" in tags and "platform" in lowered:
        categories.add("platform")
    #if "block" in lowered or "brick" in lowered:
    #    categories.add("block")
    for category in ("ladder", "water", "lava", "spring", "coin", "pipe"):
        if category in lowered:
            categories.add(category)
    if "door" in tags or ("door" in lowered and "key door" not in lowered):
        categories.add("door")
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
            "terms": tile_name_terms(description),
            "phrases": matching_phrases(description, tags),
            "categories": category_for_tile(description, tags),
        }
    return {"categories": vocabulary, "tiles": tile_concepts, "tile_descriptors": tile_descriptors}


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
    if any(
        char in present_tiles and "fading platform" in info["phrases"]
        for char, info in vocabulary["tiles"].items()
    ):
        present_categories.add("platform")
    if scene_has_platform(scene, id_to_char, vocabulary["tile_descriptors"], vocabulary["tiles"]):
        present_categories.add("platform")

    # Evaluate category alternatives explicitly while preserving a stable, human-readable result.
    required = sorted(category for category in present_categories
                 if any(category_term_present(tokens, category, term)
                     for term in vocabulary["categories"][category])
                      or (category == "powerup" and any(
                          phrase_present(tokens, phrase)
                          for info in vocabulary["tiles"].values()
                          for phrase in info["phrases"]
                          if phrase in POWERUP_COMPOUND_CONCEPTS)))
    all_categories = sorted(vocabulary["categories"])
    mentioned_categories = sorted(category for category in all_categories
                                  if any(phrase_present(tokens, term) for term in vocabulary["categories"][category])
                                  or (category == "powerup" and any(
                                      phrase_present(tokens, phrase)
                                      for info in vocabulary["tiles"].values()
                                      for phrase in info["phrases"]
                                      if phrase in POWERUP_COMPOUND_CONCEPTS)))
    unsupported_categories = sorted(set(mentioned_categories) - present_categories)
    category_matches = [
        {
            "category": category,
            "matched_terms": sorted(set(
                [term for term in vocabulary["categories"][category]
                 if phrase_present(tokens, term)]
                + ([phrase for info in vocabulary["tiles"].values()
                    for phrase in info["phrases"]
                    if category == "powerup"
                    and phrase in POWERUP_COMPOUND_CONCEPTS
                    and phrase_present(tokens, phrase)])
            )),
            "supported": category in present_categories,
        }
        for category in mentioned_categories
    ]

    category_terms = {normalize_word(term) for terms in vocabulary["categories"].values() for term in terms}
    specific_matches_by_term = {}
    for char, info in vocabulary["tiles"].items():
        # Category words such as "door" and "platform" are already scored at the
        # category level. They remain useful tile evidence, but must not be counted
        # a second time as independent precision claims.
        matched_terms = sorted(
            term for term in info["terms"]
            if normalize_word(term) not in category_terms and specific_term_present(tokens, term)
        )
        if matched_terms:
            for term in matched_terms:
                match = specific_matches_by_term.setdefault(
                    normalize_word(term),
                    {"matched_terms": [], "chars": [], "descriptions": [], "categories": set()},
                )
                if term not in match["matched_terms"]:
                    match["matched_terms"].append(term)
                match["chars"].append(char)
                match["descriptions"].append(info["description"])
                match["categories"].update(info["categories"])
        matched_phrases = sorted(
            phrase for phrase in info["phrases"] if phrase_present(tokens, phrase)
        )
        for phrase in matched_phrases:
            match = specific_matches_by_term.setdefault(
                normalize_word(phrase),
                {"matched_terms": [], "chars": [], "descriptions": [], "categories": set(), "phrase": phrase},
            )
            if phrase not in match["matched_terms"]:
                match["matched_terms"].append(phrase)
            match["chars"].append(char)
            match["descriptions"].append(info["description"])
            match["categories"].update(info["categories"])

    specific_matches = list(specific_matches_by_term.values())
    for item in specific_matches:
        item["chars"] = sorted(set(item["chars"]))
        item["descriptions"] = sorted(set(item["descriptions"]))
        item["categories"] = sorted(item["categories"])
        item["supported"] = bool(set(item["chars"]) & present_tiles)
        item["specificity_kind"] = (
            "compound" if item.get("phrase") in COMBINED_CONCEPTS
            else "modifier" if item.get("phrase") else "name"
        )
    supported_specific = [item for item in specific_matches if item["supported"]]
    unsupported_specific = [item for item in specific_matches if not item["supported"]]

    specific_category_evidence = set(
        category for item in supported_specific for category in item["categories"]
    ) | set(
        category for item in unsupported_specific for category in item["categories"]
        if category not in present_categories
    )

    generic_mentioned_categories = {
        category for category in vocabulary["categories"]
        if any(category_term_present(tokens, category, term)
               for term in vocabulary["categories"][category])
    }
    powerup_compound_mentioned = any(
        phrase_present(tokens, phrase)
        for info in vocabulary["tiles"].values()
        for phrase in info["phrases"]
        if phrase in POWERUP_COMPOUND_CONCEPTS
    )
    if powerup_compound_mentioned:
        generic_mentioned_categories.add("powerup")
    mentioned_categories = sorted(generic_mentioned_categories | specific_category_evidence)
    required = sorted(category for category in present_categories
                      if category in generic_mentioned_categories or category in specific_category_evidence)
    unsupported_categories = sorted(set(mentioned_categories) - present_categories)
    category_matches = []
    for category in mentioned_categories:
        matched_terms = {
            term for term in vocabulary["categories"][category]
            if category_term_present(tokens, category, term)
        }
        if category == "powerup":
            matched_terms.update(
                phrase for info in vocabulary["tiles"].values()
                for phrase in info["phrases"]
                if phrase in POWERUP_COMPOUND_CONCEPTS and phrase_present(tokens, phrase)
            )
        matched_terms.update(
            term for item in specific_matches
            if category in item["categories"]
            for term in item["matched_terms"]
        )
        category_matches.append({
            "category": category,
            "matched_terms": sorted(matched_terms),
            "supported": category in present_categories,
        })

    present_compounds = sorted({
        concept for info in vocabulary["tiles"].values()
        for concept in info["phrases"]
        if concept in COMBINED_CONCEPTS and any(
            char in present_tiles for char, tile_info in vocabulary["tiles"].items()
            if concept in tile_info["phrases"]
        )
    })
    has_fire_hazard = any(
        char in present_tiles
        and "hazard" in info["categories"]
        and any(word in info["description"].lower() for word in ("fire", "flame", "burner"))
        for char, info in vocabulary["tiles"].items()
    )
    if has_fire_hazard:
        present_compounds.append("fire hazard")
        present_compounds.sort()
    mentioned_compounds = sorted({
        item.get("phrase") for item in specific_matches
        if item.get("specificity_kind") == "compound"
    })
    mentioned_compounds.extend(sorted(
        concept for concept, alternatives in COMBINED_CATEGORY_CONCEPTS.items()
        if any(phrase_present(tokens, alternative) for alternative in alternatives)
    ))
    mentioned_compounds = sorted(set(mentioned_compounds))
    unsupported_compounds = sorted(set(mentioned_compounds) - set(present_compounds))
    missing_categories = sorted(present_categories - set(required))
    missing_compounds = sorted(set(present_compounds) - set(mentioned_compounds))

    # Category coverage is the main score. Specific terms are a bonus signal and do not make
    # omission of every exact enemy type look like a failure when "enemies" is accurate.
    coverage_supported_concepts = len(required) + len(set(present_compounds) & set(mentioned_compounds))
    coverage_present_concepts = len(present_categories) + len(present_compounds)
    coverage = (coverage_supported_concepts / coverage_present_concepts
                if coverage_present_concepts else 1.0)
    mentioned_count = len(mentioned_categories) + len(specific_matches)
    weighted_unsupported = len(unsupported_categories) + sum(
        1.0 if item["specificity_kind"] != "modifier" else UNSUPPORTED_SPECIFICITY_PENALTY
        for item in unsupported_specific
    )
    weighted_supported = mentioned_count - weighted_unsupported
    precision = (weighted_supported / mentioned_count
                 if mentioned_count else 1.0)
    overall = (2 * coverage * precision / (coverage + precision)
               if coverage + precision else 0.0)
    base_overall = overall
    specificity_bonus = min(0.2, 0.05 * len(supported_specific))
    overall = min(1.0, overall + specificity_bonus)
    score_breakdown = {
        "coverage_supported_categories": len(required),
        "coverage_present_categories": len(present_categories),
        "coverage_supported_compounds": len(set(present_compounds) & set(mentioned_compounds)),
        "coverage_present_compounds": len(present_compounds),
        "coverage_supported_concepts": coverage_supported_concepts,
        "coverage_present_concepts": coverage_present_concepts,
        "precision_supported_mentions": round(weighted_supported, 6),
        "precision_total_mentions": mentioned_count,
        "precision_unsupported_mentions": round(weighted_unsupported, 6),
        "precision_full_unsupported_mentions": len(unsupported_categories) + sum(
            1 for item in unsupported_specific if item["specificity_kind"] != "modifier"
        ),
        "precision_modifier_penalty": round(sum(
            UNSUPPORTED_SPECIFICITY_PENALTY for item in unsupported_specific
            if item["specificity_kind"] == "modifier"
        ), 6),
        "precision_category_mentions": len(mentioned_categories),
        "precision_specific_tile_mentions": len(specific_matches),
        "precision_supported_categories": len(mentioned_categories) - len(unsupported_categories),
        "precision_supported_specific_tiles": len(supported_specific),
        "precision_specific_mentions_exclude_category_words": True,
        "unsupported_specificity_penalty": UNSUPPORTED_SPECIFICITY_PENALTY,
        "base_overall": round(base_overall, 6),
        "specificity_bonus": round(specificity_bonus, 6),
        "specificity_bonus_formula": "min(0.2, 0.05 * supported specific concepts)",
        "coverage_formula": "supported present categories / present categories",
        "precision_formula": "supported mentions / total recognized mentions",
        "overall_formula": "2 * coverage * precision / (coverage + precision)",
    }
    return {
        "coverage": round(coverage, 6),
        "precision": round(max(0.0, precision), 6),
        "overall": round(overall, 6),
        "base_overall": round(base_overall, 6),
        "specificity_bonus": round(specificity_bonus, 6),
        "present_categories": sorted(present_categories),
        "mentioned_categories": mentioned_categories,
        "unsupported_categories": unsupported_categories,
        "category_matches": category_matches,
        "present_compound_concepts": present_compounds,
        "mentioned_compound_concepts": mentioned_compounds,
        "unsupported_compound_concepts": unsupported_compounds,
        "missing_categories": missing_categories,
        "missing_compound_concepts": missing_compounds,
        "supported_specific_tiles": supported_specific,
        "unsupported_specific_tiles": unsupported_specific,
        "scene_tile_counts": dict(present),
        "score_breakdown": score_breakdown,
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
            scored_entries.append({
                "entry_index": entry_index,
                "scene": entry["scene"],
                "scores": caption_scores,
            })

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