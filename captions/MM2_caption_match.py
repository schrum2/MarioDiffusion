"""Caption comparison for the deterministic MM captions from
MarioMaker_create_ascii_captions.py.

Unlike the fixed keyword list in caption_match.py, MM captions mention whatever
tiles are present ("Two goombas. A few coins. A blob of coins."), so topics are
derived from the tileset: each entity name is a topic, and "a blob of X" is a
separate topic grouped under the same entity (topic_entity). Metadata phrases
(style/theme/difficulty/tags) aren't derivable from a bare scene, so they're
ignored. The final score divides by the topics either caption mentions, not the
whole vocabulary, so shared absences don't drown out real differences.
"""
import os
import sys

# Run directly, python puts the captions dir on sys.path where captions/util.py
# shadows the root util package. Swap it for the repo root.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]
sys.path.insert(0, os.path.dirname(_here))

import util.common_settings as common_settings
from MarioMaker_create_ascii_captions import (
    build_id_to_char, get_char_names, get_tile_categories, assign_caption,
    pluralize)

QUANTITY_TERMS = ["one", "two", "a few", "several", "many", "a ton of"]

# CAPTION_METADATA_FIELDS suffixes. Tags have no suffix and just fail lookup.
METADATA_SUFFIXES = (" style", " theme", " difficulty")


def build_name_lookup(tileset_path):
    """Maps every singular/plural entity name to its canonical singular form.
    Ground chars are excluded (the floor phrase covers them)."""
    char_names = get_char_names(tileset_path)
    _, _, ground_chars = get_tile_categories(tileset_path)
    lookup = {}
    for char, name in char_names.items():
        if char in ground_chars:
            continue
        # Stored without trailing periods, which the phrase split strips off.
        base = name.lower().rstrip(".")
        lookup[base] = base
        lookup[pluralize(base).rstrip(".")] = base
    return lookup


NAME_LOOKUP = build_name_lookup(common_settings.MM2_TILESET)


def phrase_topic(phrase, name_lookup=None):
    """Sort one caption phrase into its topic.

    Returns (topic, kind) where kind is one of "count", "blob", "ground" or
    "metadata". Unrecognized phrases (tags, LLM wording) return (None, None).
    """
    if name_lookup is None:
        name_lookup = NAME_LOOKUP
    p = phrase.strip().lower().rstrip(".")
    if not p:
        return None, None

    if p == "full ground floor" or p == "scattered ground" or p.startswith("ground floor with"):
        return "ground", "ground"

    if p.startswith("a blob of "):
        name = name_lookup.get(p[len("a blob of "):])
        if name:
            return f"blob of {name}", "blob"
        return None, None

    for qty in sorted(QUANTITY_TERMS, key=len, reverse=True):
        if p.startswith(qty + " "):
            name = name_lookup.get(p[len(qty) + 1:])
            if name:
                return name, "count"
            return None, None

    for suffix in METADATA_SUFFIXES:
        if p.endswith(suffix):
            return suffix.strip(), "metadata"

    return None, None


def topic_entity(topic):
    """The entity a topic is about: "goomba" and "blob of goomba" both group
    under "goomba" (the grouping hook for issue #35)."""
    if topic.startswith("blob of "):
        return topic[len("blob of "):]
    return topic


def extract_topics(caption, name_lookup=None, include_metadata=False):
    """Maps topic -> normalized phrase for every scorable phrase in a caption."""
    topics = {}
    for raw in caption.split("."):
        topic, kind = phrase_topic(raw, name_lookup)
        if topic is None:
            continue
        if kind == "metadata" and not include_metadata:
            continue
        topics.setdefault(topic, raw.strip().lower().rstrip("."))
    return topics


def group_phrases(caption, name_lookup=None):
    """Groups a caption's phrases by the entity they describe: entity -> list
    of phrases. The ground phrase groups under "ground"."""
    groups = {}
    for raw in caption.split("."):
        topic, kind = phrase_topic(raw, name_lookup)
        if topic is None or kind == "metadata":
            continue
        groups.setdefault(topic_entity(topic), []).append(raw.strip().lower().rstrip("."))
    return groups


def quantity_score(phrase1, phrase2, debug=False):
    def find_quantity(phrase):
        for term in sorted(QUANTITY_TERMS, key=len, reverse=True):
            if term in phrase:
                return term
        return None

    qty1 = find_quantity(phrase1)
    qty2 = find_quantity(phrase2)
    if qty1 and qty2:
        diff = abs(QUANTITY_TERMS.index(qty1) - QUANTITY_TERMS.index(qty2))
        score = 1.0 - (diff / (len(QUANTITY_TERMS) - 1))
        if debug:
            print(f"[Quantity] '{qty1}' vs '{qty2}' -> {score:.2f}")
        return score
    if debug:
        print("[Quantity] missing quantity term, partial score 0.1")
    return 0.1


def compare_captions(correct_caption, generated_caption, debug=False, return_matches=False, name_lookup=None):
    """Score generated_caption against correct_caption over the topics that
    either caption mentions. Same return shape as caption_match.compare_captions."""
    correct = extract_topics(correct_caption, name_lookup)
    generated = extract_topics(generated_caption, name_lookup)

    topics = sorted(set(correct) | set(generated))
    exact_matches = []
    partial_matches = []
    excess_phrases = []

    if not topics:
        # Neither caption says anything scorable: agreement.
        if return_matches:
            return 1.0, exact_matches, partial_matches, excess_phrases
        return 1.0

    total_score = 0.0
    for topic in topics:
        c = correct.get(topic)
        g = generated.get(topic)
        if debug:
            print(f"[Topic: {topic}] Correct: {c} | Generated: {g}")

        if c is None or g is None:
            total_score += -1.0
            if g is not None:
                excess_phrases.append(g)
        elif c == g:
            total_score += 1.0
            exact_matches.append(g)
        elif topic.startswith("blob of "):
            # Blobs carry no quantity, so both mentioning it is a match.
            total_score += 1.0
            exact_matches.append(g)
        elif c.startswith("ground floor with") and g.startswith("ground floor with"):
            total_score += quantity_score(c, g, debug=debug)
            partial_matches.append(g)
        elif topic == "ground":
            # Different floor categories (full/gaps/scattered): topic overlap only.
            total_score += 0.1
            partial_matches.append(g)
        else:
            total_score += quantity_score(c, g, debug=debug)
            partial_matches.append(g)

    final_score = total_score / len(topics)
    if debug:
        print(f"--- Final score: {final_score:.4f} over {len(topics)} topics ---")

    if return_matches:
        return final_score, exact_matches, partial_matches, excess_phrases
    return final_score


def caption_tools(tileset_path=common_settings.MM2_TILESET):
    """(assign_fn, compare_fn) for MM2 scenes, for the hooks in
    calculate_caption_score_and_samples. assign_fn takes just a scene."""
    id_to_char = build_id_to_char(tileset_path)
    char_names = get_char_names(tileset_path)
    _, _, ground_chars = get_tile_categories(tileset_path)
    name_lookup = build_name_lookup(tileset_path)

    def assign_fn(scene):
        return assign_caption(scene, id_to_char, char_names, ground_chars)

    def compare_fn(correct_caption, generated_caption, **kwargs):
        return compare_captions(correct_caption, generated_caption,
                                name_lookup=name_lookup, **kwargs)

    return assign_fn, compare_fn


if __name__ == '__main__':
    ref = "Full ground floor. Two goombas. A few coins. A blob of coins."
    gen = "Full ground floor. One goomba. Several coins. One thwomp."

    score = compare_captions(ref, gen, debug=True)
    print(f"Should be: {ref}")
    print(f"  but was: {gen}")
    print(f"Score: {score}")
