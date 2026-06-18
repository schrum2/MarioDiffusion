"""Build a multi-caption dataset from an existing single-caption dataset.

Each sample's structured "caption" is split into its period-separated phrases and several
random re-orderings are stored as "caption1", "caption2", ... alongside the unchanged
"caption". The result follows the multi-caption convention used by llm_ascii_to_caption.py
(scene + caption + caption1..captionN), but here the alternatives are phrase shuffles of a
single caption rather than distinct descriptions.

This exists as an equivalence test for the --multiple_captions training option: training on
the generated dataset with --multiple_captions selects a random phrase ordering per access,
which should behave like the existing phrase-shuffle augmentation (--augment). If the two
training runs agree, the multiple-caption plumbing is wired up correctly.
"""

import argparse
import json
import math
import random


def make_caption_orderings(caption, num_captions):
    """Return num_captions versions of caption: the original followed by random phrase orderings.

    The original caption is kept verbatim as the first element (so the stored "caption" field is
    unchanged and other tools keep reading the canonical caption). The remaining elements are
    random orderings of the same phrases, kept distinct from one another when the phrase count
    allows it; once the distinct permutations run out, repeats fill the remaining slots.
    """
    body = caption[:-1] if caption.endswith(".") else caption
    phrases = [p for p in body.split(". ") if p]

    orderings = [caption]
    seen = {tuple(phrases)}
    # Cap distinct orderings at the number of permutations so single-phrase (or short) captions
    # don't loop forever trying to find new arrangements that don't exist.
    max_distinct = math.factorial(len(phrases)) if len(phrases) <= 8 else None

    while len(orderings) < num_captions:
        shuffled = phrases[:]
        random.shuffle(shuffled)
        key = tuple(shuffled)
        exhausted = max_distinct is not None and len(seen) >= max_distinct
        if key in seen and not exhausted:
            continue
        seen.add(key)
        orderings.append(". ".join(shuffled) + "." if shuffled else caption)

    return orderings


def build_multiple_caption_data(data, num_captions):
    new_data = []
    for sample in data:
        if "caption" not in sample:
            raise ValueError(f"Sample is missing a 'caption' field: {sample.keys()}")
        orderings = make_caption_orderings(sample["caption"], num_captions)
        # Preserve every non-caption field (scene, prompt, ...) and overwrite the caption fields.
        new_sample = {k: v for k, v in sample.items() if not _is_caption_key(k)}
        new_sample["caption"] = orderings[0]
        for i, ordering in enumerate(orderings[1:], start=1):
            new_sample[f"caption{i}"] = ordering
        new_data.append(new_sample)
    return new_data


def _is_caption_key(key):
    return key == "caption" or (key.startswith("caption") and key[len("caption"):].isdigit())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the source single-caption dataset JSON")
    parser.add_argument("--output", required=True, help="Path to write the multi-caption dataset JSON")
    parser.add_argument("--num_captions", type=int, default=5,
                        help="Total number of caption fields per sample (caption + caption1..captionN-1). Default: 5")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible orderings")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_captions < 1:
        raise ValueError("--num_captions must be at least 1")

    random.seed(args.seed)

    with open(args.input, "r") as f:
        data = json.load(f)

    new_data = build_multiple_caption_data(data, args.num_captions)

    with open(args.output, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Wrote {len(new_data)} samples with {args.num_captions} caption field(s) each to {args.output}")


if __name__ == "__main__":
    main()
