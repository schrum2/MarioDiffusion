"""Caption comparison for the deterministic MMLV captions from MMLV_create_ascii_captions.py.

Phrases come from a bank built out of the captioner's name tables and the tileset. As in the MM2
scorer, the score divides by the topics either caption mentions. Disagreeing about locations, ladder
positions or spike pit lengths costs up to QUALIFIER_WEIGHT of a topic's score.

Command line, from the repo root:
    python captions/MMLV_caption_match.py --check Game_MMLV/DATA/foo.json
    python captions/MMLV_caption_match.py --self_check
    python captions/MMLV_caption_match.py --list_phrases
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, namedtuple

# Run directly, python puts the captions dir on sys.path where captions/util.py
# shadows the root util package. Swap it for the repo root.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]
_repo_root = os.path.dirname(_here)
sys.path.insert(0, _repo_root)

import util.common_settings as common_settings
from captions.util import describe_quantity, extract_tileset
from MMLV_create_ascii_captions import (
    BREAKABLE_BLOCK_NAMES, CAPTION_VERSION_KEY, CAPTIONER_VERSION, DISAPPEARING_BLOCK_NAMES,
    ENEMY_NAMES, HAZARD_NAMES, IRREGULAR_CLUSTER_NAMES, LAVA_CHAR, LAVA_NAMES, LOOSE_BLOCK_NAMES,
    MMLV_TILE_TYPES, MOVING_PLATFORM_NAMES, PLATFORM_NAMES, POWERUP_NAMES,
    RECTANGULAR_CLUSTER_NAMES, SPIKE_CHAR, SPIKE_NAMES, TOWER_NAMES, _pair, assign_caption,
    tile_structure_names)

# The quantity words in order, taken from describe_quantity so its thresholds have one home.
# Each word's typical count is what lets phrases at different locations be added up.
_COUNTS_BY_TERM = {}
for _count in range(1, 31):
    _COUNTS_BY_TERM.setdefault(describe_quantity(_count), []).append(_count)
QUANTITY_TERMS = list(_COUNTS_BY_TERM)
QUANTITY_COUNTS = {term: statistics.median(counts) for term, counts in _COUNTS_BY_TERM.items()}

WATER_TERMS = ["some", "half", "mostly", "all"]
LOCATIONS = ("at left", "at center", "at right", "across")
# Where each third of the scene sits, for measuring how far apart two locations are
LOCATION_POSITIONS = {"at left": 0, "at center": 1, "at right": 2}
# "across" spans most of the scene, so it partly overlaps every third
ACROSS_CLOSENESS = 0.5
LADDER_POSITIONS = ("at top", "at bottom", "in the middle")
STRIP_LENGTHS = ("short", "long", "full")

# Most of a shared topic's score that disagreeing about where it is can cost
QUALIFIER_WEIGHT = 0.25
# Score for describing a floor, ceiling or wall the wrong way: the topic overlaps and nothing else
PARTIAL_CREDIT = 0.1

_QUANTITY_PATTERN = "|".join(re.escape(term) for term in QUANTITY_TERMS)
_FLOOR_GAPS = re.compile(rf"^floor with ({_QUANTITY_PATTERN}) gaps?$")
_FLOOR_CHUNKS = re.compile(rf"^giant gap with ({_QUANTITY_PATTERN}) chunks? of floor$")
_CEILING_GAPS = re.compile(rf"^ceiling with ({_QUANTITY_PATTERN}) gaps?$")

PhraseBank = namedtuple("PhraseBank", "counted strips absences")
"""counted: every singular/plural name that follows a quantity, mapped to its singular topic.
strips: the counted topics that also carry a length word ("one long spike pit").
absences: everything that may follow "no " under --describe_absence."""

ParsedPhrase = namedtuple("ParsedPhrase", "topic kind quantity variant location text")
"""topic: what the phrase is about, None for phrases that aren't scored.
kind: "count", "layout" (floor, ceiling, walls), "scale" (water), or "absence"/"metadata".
quantity: the quantity word, or the water word.
variant: which form of the topic ("gaps", "perforated", "at top", "long").
location: "at left", "across", or None.
text: the phrase as normalized for comparison."""


def build_phrase_bank(tileset_path=common_settings.MMLV_TILESET):
    """Every phrase the captioner can write for this tileset. A tile the tileset doesn't have
    contributes nothing, so a phrase about it reads as unknown instead of silently scoring."""
    _, _, char_to_id, tile_descriptors = extract_tileset(tileset_path)
    counted = {}
    strips = set()
    absences = {"floor", "ceiling", "left wall", "right wall", "water"}

    def add(names, count_only=False):
        singular, plural = names
        counted[singular] = singular
        counted[plural] = singular
        if not count_only:
            absences.add(plural)

    # An enemy, power-up or hazard the table doesn't name is counted under its family's name
    for table, descriptor, family in ((ENEMY_NAMES, "enemy", ("enemy", "enemies")),
                                      (POWERUP_NAMES, "powerup", _pair("powerup")),
                                      (HAZARD_NAMES, "hazard", _pair("hazard"))):
        for char, names in table.items():
            if char in char_to_id:
                add(names, count_only=True)
        absences.add(family[1])
        strip_chars = (SPIKE_CHAR, LAVA_CHAR)
        if any(descriptor in value and char not in table and char not in strip_chars
               for char, value in tile_descriptors.items()):
            add(family, count_only=True)

    if SPIKE_CHAR in char_to_id:
        absences.add("spikes")
        for kind, names in SPIKE_NAMES.items():
            add(names, count_only=True)
            if kind == "pit":
                strips.add(names[0])
    if LAVA_CHAR in char_to_id:
        add(LAVA_NAMES)
        strips.add(LAVA_NAMES[0])

    for chars, names, _multi_tile in MMLV_TILE_TYPES:
        if any(char in char_to_id for char in chars):
            add(names)

    for char, names in MOVING_PLATFORM_NAMES.items():
        if char in char_to_id:
            add(names, count_only=True)
    absences.add(MOVING_PLATFORM_NAMES["M"][1])

    if "A" in char_to_id:
        add(DISAPPEARING_BLOCK_NAMES)
    for names in (PLATFORM_NAMES, LOOSE_BLOCK_NAMES, TOWER_NAMES, RECTANGULAR_CLUSTER_NAMES,
                  IRREGULAR_CLUSTER_NAMES):
        add(names)
    for names in tile_structure_names(*BREAKABLE_BLOCK_NAMES):
        add(names, count_only=True)
    absences.add(BREAKABLE_BLOCK_NAMES[1])
    absences.add("ladders")

    return PhraseBank(counted, strips, absences)


PHRASE_BANK = build_phrase_bank()


def build_topic_keywords(bank=PHRASE_BANK):
    """Topics that the ascii data browser colors caption phrases by, longest first so "breakable
    block cluster" is matched before "breakable block". A plural that doesn't contain its singular
    ("batteries") needs its own keyword to be matched at all."""
    topics = {"floor", "ceiling", "left wall", "right wall", "water", "ladder"}
    for form, singular in bank.counted.items():
        if form == singular or singular not in form:
            topics.add(form)
    return sorted(topics, key=len, reverse=True)


TOPIC_KEYWORDS = build_topic_keywords()


def _parse_layout(p):
    """Floor, ceiling, wall and water phrases, or None. Their wording has to match
    describe_floor_occupancy, analyze_ceiling, find_water_caption and assign_caption's wall
    section; --self_check catches it if they drift apart."""
    def layout(topic, variant, quantity=None):
        return ParsedPhrase(topic, "layout", quantity, variant, None, p)

    if p == "full floor":
        return layout("floor", "full")
    if p == "full ceiling":
        return layout("ceiling", "full")
    for pattern, topic, variant in ((_FLOOR_GAPS, "floor", "gaps"), (_FLOOR_CHUNKS, "floor", "chunks"),
                                    (_CEILING_GAPS, "ceiling", "gaps")):
        match = pattern.match(p)
        if match:
            return layout(topic, variant, match.group(1))
    for side in ("left wall", "right wall"):
        if p == side:
            return layout(side, "solid")
        if p == "perforated " + side:
            return layout(side, "perforated")
    if p.endswith(" water") and p[:-len(" water")] in WATER_TERMS:
        return ParsedPhrase("water", "scale", p[:-len(" water")], None, None, p)
    return None


def _parse_counted(quantity, rest, p, bank):
    """The name after a quantity word, with its location, ladder position or length. None if the
    name isn't in the bank."""
    location = None
    for where in LOCATIONS:
        if rest.endswith(" " + where):
            location = where
            rest = rest[:-len(where) - 1]
            break

    for position in LADDER_POSITIONS:
        if rest in (f"ladder {position}", f"ladders {position}"):
            return ParsedPhrase("ladder", "count", quantity, position, location, p)
    if rest in ("full height ladder", "full height ladders"):
        return ParsedPhrase("ladder", "count", quantity, "full height", location, p)

    first, _, remainder = rest.partition(" ")
    if first in STRIP_LENGTHS and bank.counted.get(remainder) in bank.strips:
        return ParsedPhrase(bank.counted[remainder], "count", quantity, first, location, p)

    topic = bank.counted.get(rest)
    if topic is None:
        return None
    return ParsedPhrase(topic, "count", quantity, None, location, p)


def parse_phrase(phrase, bank=None):
    """Sort one caption phrase into a ParsedPhrase, or None if the bank doesn't know it."""
    bank = PHRASE_BANK if bank is None else bank
    p = phrase.strip().lower().rstrip(".")
    if not p:
        return None
    if p.startswith("no ") and p[len("no "):] in bank.absences:
        return ParsedPhrase(None, "absence", None, None, None, p)
    # Not derivable from a bare scene, so generated levels could never match them.
    if p.startswith(("entrance direction ", "exit direction ")):
        return ParsedPhrase(None, "metadata", None, None, None, p)

    layout = _parse_layout(p)
    if layout is not None:
        return layout
    for term in sorted(QUANTITY_TERMS, key=len, reverse=True):
        if p.startswith(term + " "):
            return _parse_counted(term, p[len(term) + 1:], p, bank)
    return None


def split_phrases(caption):
    return [p.strip() for p in caption.split(".") if p.strip()]


def unknown_phrases(caption, bank=None):
    """The phrases of a caption the bank doesn't recognize. They aren't scored, so a stale bank or
    a caption from an LLM has to be noticed some other way: see check_dataset."""
    return [p for p in split_phrases(caption) if parse_phrase(p, bank) is None]


def extract_topics(caption, bank=None):
    """Maps topic -> list of ParsedPhrase for every scorable phrase in a caption. A topic can
    have several, one per location or ladder position."""
    topics = {}
    for raw in split_phrases(caption):
        parsed = parse_phrase(raw, bank)
        if parsed is not None and parsed.topic is not None:
            topics.setdefault(parsed.topic, []).append(parsed)
    return topics


def extract_phrases(caption, debug=False, bank=None):
    """Maps topic -> the caption's phrases about it, as extract_phrases does in the other games."""
    phrases = {topic: ". ".join(p.text for p in parsed) for topic, parsed in extract_topics(caption, bank).items()}
    if debug:
        for topic, phrase in phrases.items():
            print(f"[Extract] Topic '{topic}': found phrase(s) '{phrase}'")
    return phrases


def quantity_score(term1, term2, debug=False):
    diff = abs(QUANTITY_TERMS.index(term1) - QUANTITY_TERMS.index(term2))
    score = 1.0 - (diff / (len(QUANTITY_TERMS) - 1))
    if debug:
        print(f"[Quantity] '{term1}' vs '{term2}' -> {score:.2f}")
    return score


def _total_quantity(parsed):
    """The quantity word for all of a topic's phrases together."""
    return describe_quantity(round(sum(QUANTITY_COUNTS[p.quantity] for p in parsed)))


def _variant_factor(correct, generated):
    """Multiplier for agreement on a topic's ladder positions or spike pit lengths. Only captions
    that both give this detail can disagree about it."""
    c = {p.variant for p in correct} - {None}
    g = {p.variant for p in generated} - {None}
    if not c or not g:
        return 1.0
    return (1.0 - QUALIFIER_WEIGHT) + QUALIFIER_WEIGHT * len(c & g) / len(c | g)


def _location_closeness(location1, location2):
    """1.0 for the same location, 0.0 for opposite sides of the scene."""
    if location1 == location2:
        return 1.0
    if "across" in (location1, location2):
        return ACROSS_CLOSENESS
    return 1.0 - abs(LOCATION_POSITIONS[location1] - LOCATION_POSITIONS[location2]) / 2


def _location_factor(correct, generated):
    """Multiplier for agreement on a topic's locations. Each location in either caption is judged
    by the closest location in the other caption. Only captions that both give locations can
    disagree about them."""
    c = {p.location for p in correct} - {None}
    g = {p.location for p in generated} - {None}
    if not c or not g:
        return 1.0
    closeness = ([max(_location_closeness(a, b) for b in g) for a in c]
                 + [max(_location_closeness(a, b) for b in c) for a in g])
    return (1.0 - QUALIFIER_WEIGHT) + QUALIFIER_WEIGHT * sum(closeness) / len(closeness)


def _score_topic(correct, generated, debug=False):
    """Score, in [0, 1], for a topic that both captions mention."""
    c, g = correct[0], generated[0]
    if c.kind == "scale":
        return 1.0 - abs(WATER_TERMS.index(c.quantity) - WATER_TERMS.index(g.quantity)) / (len(WATER_TERMS) - 1)
    if c.kind == "layout":
        if c.variant != g.variant:
            return PARTIAL_CREDIT
        return quantity_score(c.quantity, g.quantity, debug=debug) if c.quantity else 1.0
    return (quantity_score(_total_quantity(correct), _total_quantity(generated), debug=debug)
            * _variant_factor(correct, generated)
            * _location_factor(correct, generated))


def compare_captions(correct_caption, generated_caption, debug=False, return_matches=False, bank=None):
    """Score generated_caption against correct_caption over the topics that either caption
    mentions. Same return shape as caption_match.compare_captions."""
    correct = extract_topics(correct_caption, bank)
    generated = extract_topics(generated_caption, bank)

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
        g_text = ". ".join(p.text for p in g) if g else None
        if debug:
            c_text = ". ".join(p.text for p in c) if c else None
            print(f"[Topic: {topic}] Correct: {c_text} | Generated: {g_text}")

        if c is None or g is None:
            total_score += -1.0
            if g is not None:
                excess_phrases.append(g_text)
            continue

        topic_score = _score_topic(c, g, debug=debug)
        total_score += topic_score
        (exact_matches if topic_score == 1.0 else partial_matches).append(g_text)

    final_score = total_score / len(topics)
    if debug:
        print(f"--- Final score: {final_score:.4f} over {len(topics)} topics ---")

    if return_matches:
        return final_score, exact_matches, partial_matches, excess_phrases
    return final_score


def caption_tools(tileset_path=common_settings.MMLV_TILESET, describe_locations=True, describe_absence=False):
    """(assign_fn, compare_fn) for MMLV scenes, for the hooks in
    calculate_caption_score_and_samples. assign_fn takes just a scene."""
    _, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)
    bank = build_phrase_bank(tileset_path)

    def assign_fn(scene):
        return assign_caption(scene, id_to_char, char_to_id, tile_descriptors,
                              describe_locations, describe_absence)

    def compare_fn(correct_caption, generated_caption, **kwargs):
        return compare_captions(correct_caption, generated_caption, bank=bank, **kwargs)

    return assign_fn, compare_fn


def process_scene_segments(scene, segment_width, prompt, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence, verbose=False):
    """
    Process a scene by partitioning it into segments, assigning captions, and computing comparison scores.

    Returns:
        tuple: A tuple containing the average comparison score, captions for each segment, and scores for each segment.
    """
    segments = [
        [row[i:i+segment_width] for row in scene]
        for i in range(0, len(scene[0]), segment_width)
    ]

    segment_scores = []
    segment_captions = []
    for idx, segment in enumerate(segments):
        segment_caption = assign_caption(segment, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence)
        segment_score = compare_captions(prompt, segment_caption)
        segment_scores.append(segment_score)
        segment_captions.append(segment_caption)

        if verbose:
            print(f"Segment {idx + 1} caption: {segment_caption}")
            print(f"Segment {idx + 1} comparison score: {segment_score}")

    average_score = sum(segment_scores) / len(segment_scores) if segment_scores else 0

    if verbose:
        print(f"Average comparison score across all segments: {average_score}")

    return average_score, segment_captions, segment_scores


def deterministic_caption(entry):
    """An entry's deterministic caption from either --caption-mode, or None."""
    keyed = entry.get("deterministic_captions")
    if isinstance(keyed, list) and keyed:
        return keyed[0]
    caption = entry.get("caption")
    return caption if isinstance(caption, str) and caption else None


def _recaption_flags(caption):
    """The describe_locations/describe_absence settings a caption was written with, worked out
    from its wording. A caption with no location words reads the same either way."""
    phrases = [p.rstrip(".") for p in split_phrases(caption.lower())]
    return (any(p.endswith(LOCATIONS) for p in phrases),
            any(p.startswith("no ") for p in phrases))


def _recaption_matches(entry, caption, tools):
    id_to_char, char_to_id, tile_descriptors = tools
    scene = entry["scene"]
    scene = scene.tolist() if hasattr(scene, "tolist") else scene
    locations, absence = _recaption_flags(caption)
    current = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, locations, absence,
                             entry.get("data"))
    return current.strip() == caption.strip(), current


def check_dataset(entries, tileset_path=common_settings.MMLV_TILESET, sample=500, seed=0):
    """Is this dataset's text safe to score against levels described by the current captioner?

    Looks at three things: which captioner version the entries record, whether the bank recognizes
    their phrases, and (for entries with scenes) whether the current captioner still writes the
    same caption for the same scene. That last one is the real test, since it also catches a
    captioner change nobody bumped CAPTIONER_VERSION for. A file with no scenes has only its
    recorded version to go on.

    sample: how many scenes to re-caption, 0 for all of them.
    Returns a report dict with "status": "compatible", "unverified" or "incompatible".
    """
    _, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)
    bank = build_phrase_bank(tileset_path)

    versions = Counter()
    unknown = Counter()
    captioned = []
    for entry in entries:
        caption = deterministic_caption(entry)
        if caption is None:
            continue
        versions[entry.get(CAPTION_VERSION_KEY)] += 1
        unknown.update(unknown_phrases(caption, bank))
        if entry.get("scene") is not None:
            captioned.append((entry, caption))

    if sample and len(captioned) > sample:
        captioned = random.Random(seed).sample(captioned, sample)
    tools = (id_to_char, char_to_id, tile_descriptors)
    matched = 0
    differences = []
    for entry, caption in captioned:
        same, recaptioned = _recaption_matches(entry, caption, tools)
        matched += same
        if not same and len(differences) < 5:
            differences.append({"stored": caption, "current": recaptioned})

    total = sum(versions.values())
    if unknown or (captioned and matched < len(captioned)):
        status = "incompatible"
    elif captioned or versions[CAPTIONER_VERSION] == total > 0:
        status = "compatible"
    else:
        status = "unverified"

    return {
        "status": status,
        "captions": total,
        "unstamped": versions[None],
        "versions": {version: count for version, count in versions.items() if version is not None},
        "current_version": CAPTIONER_VERSION,
        "unknown_phrases": dict(unknown.most_common(20)),
        "unknown_phrase_count": sum(unknown.values()),
        "recaptioned": len(captioned),
        "recaption_matches": matched,
        "recaption_differences": differences,
    }


def describe_report(report):
    """Lines of text explaining a check_dataset report."""
    lines = [f"MMLV caption check: {report['status'].upper()} ({report['captions']} deterministic captions)"]
    lines.append(f"  Current captioner: version {report['current_version']}")
    if report["unstamped"]:
        lines.append(f"  {report['unstamped']} captions don't record which captioner version wrote them.")
    for version, count in report["versions"].items():
        lines.append(f"  {count} captions from version {version}")
    if report["recaptioned"]:
        lines.append(f"  Current captioner reproduces {report['recaption_matches']} of {report['recaptioned']} sampled captions exactly.")
    else:
        lines.append("  No scenes to re-caption, so the captioner can't be tested against this file.")
    for difference in report["recaption_differences"]:
        lines.append(f"    stored:  {difference['stored']}")
        lines.append(f"    current: {difference['current']}")
    if report["unknown_phrase_count"]:
        lines.append(f"  {report['unknown_phrase_count']} phrases aren't in the phrase bank and won't be scored: {report['unknown_phrases']}")
    if report["status"] == "incompatible":
        lines.append("  Scores on this data are unreliable: its captions and generated levels' captions don't speak the same language. Re-caption it with MMLV_create_ascii_captions.py.")
    elif report["status"] == "unverified":
        lines.append("  Scores may be unreliable: nothing shows these captions came from the current captioner.")
    return lines


def self_check(tileset_path=common_settings.MMLV_TILESET, scenes=300, seed=0):
    """Run the captioner over synthetic scenes containing every tile, and report any phrase the
    bank can't read, any caption that doesn't score 1.0 against itself, and any bank topic the
    captioner never produced. Run this after changing the captioner or the tileset. Returns the
    number of problems found."""
    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)
    bank = build_phrase_bank(tileset_path)
    rng = random.Random(seed)
    tile_ids = list(id_to_char)
    empty = char_to_id["-"]
    null = char_to_id["@"]
    solid = char_to_id["#"]
    unknown = Counter()
    failures = []
    seen = set()

    def blank():
        return [[null] * 16 for _ in range(2)] + [[empty] * 16 for _ in range(14)]

    test_scenes = [blank()]
    floor = blank()
    floor[15] = [solid] * 16
    floor[3] = [solid] * 16
    test_scenes.append(floor)
    gaps = blank()
    gaps[15] = [solid] * 4 + [empty] * 2 + [solid] * 10
    gaps[2] = [solid] * 4 + [empty] * 2 + [solid] * 10
    test_scenes.append(gaps)
    tower = blank()
    tower[15] = [solid] * 16
    for row in range(11, 15):
        tower[row][5] = solid
    test_scenes.append(tower)
    for tile in tile_ids:  # every tile, filling the scene and alone
        filled = blank()
        for row in range(2, 16):
            filled[row] = [tile] * 16
        lone = blank()
        lone[9][8] = tile
        test_scenes += [filled, lone]
    for _ in range(scenes):
        scene = blank()
        for _ in range(rng.randint(1, 40)):
            tile = rng.choice(tile_ids)
            row, col = rng.randrange(2, 16), rng.randrange(16)
            for offset in range(rng.randint(1, 6)):
                if col + offset < 16:
                    scene[row][col + offset] = tile
        test_scenes.append(scene)

    # Ceilings are only described when the scene's exit direction is known
    directions = {"entrance_direction": "LEFT", "exit_direction": "RIGHT"}
    for scene in test_scenes:
        for locations in (True, False):
            for absence in (True, False):
                for data in (None, directions):
                    caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, locations, absence, data)
                    seen.update(split_phrases(caption))
                    unknown.update(unknown_phrases(caption, bank))
                    score = compare_captions(caption, caption, bank=bank)
                    if score != 1.0 and len(failures) < 5:
                        failures.append((score, caption))

    produced = {parsed.topic for parsed in (parse_phrase(p, bank) for p in seen) if parsed is not None}
    unused = (set(bank.counted.values()) | {"floor", "ceiling", "left wall", "right wall", "water", "ladder"}) - produced

    print(f"Self check: {len(test_scenes)} scenes produced {len(seen)} distinct phrases.")
    if unknown:
        print(f"Phrases the bank can't read: {dict(unknown)}")
    if unused:
        print(f"Topics in the bank the captioner never produced: {sorted(unused)}")
    for score, caption in failures:
        print(f"Scored {score} against itself: {caption}")
    problems = len(unknown) + len(unused) + len(failures)
    print("No problems found." if not problems else f"{problems} problems found.")
    return problems


def parse_args():
    parser = argparse.ArgumentParser(description="Score MMLV captions, and check MMLV datasets against the current captioner")
    parser.add_argument("--tileset", default=common_settings.MMLV_TILESET, help="Descriptions of individual tile types")
    parser.add_argument("--check", type=str, default=None, help="Dataset json to check against the current captioner")
    parser.add_argument("--sample", type=int, default=500, help="With --check, how many scenes to re-caption (0 for all)")
    parser.add_argument("--self_check", action="store_true", help="Check that the phrase bank can read everything the captioner writes")
    parser.add_argument("--list_phrases", action="store_true", help="Print the phrase bank")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.self_check:
        sys.exit(1 if self_check(args.tileset) else 0)
    elif args.list_phrases:
        phrase_bank = build_phrase_bank(args.tileset)
        print("Counted names:", sorted(set(phrase_bank.counted)))
        print("Also take a length:", sorted(phrase_bank.strips))
        print("Can follow 'no':", sorted(phrase_bank.absences))
        print("Quantities:", QUANTITY_TERMS)
    elif args.check:
        with open(args.check, "r", encoding="utf-8") as f:
            check_report = check_dataset(json.load(f), args.tileset, sample=args.sample)
        print("\n".join(describe_report(check_report)))
    else:
        ref = "full floor. two mets. one platform at left. one platform at right."
        gen = "floor with one gap. a few mets. two platforms at left. one spine."

        score = compare_captions(ref, gen, debug=True)
        print(f"Should be: {ref}")
        print(f"  but was: {gen}")
        print(f"Score: {score}")
