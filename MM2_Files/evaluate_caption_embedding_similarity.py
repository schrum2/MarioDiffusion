"""Compare the captions an LLM gives one scene, using the text encoders the
diffusion models train with.

Usage:
    python MM2_Files/evaluate_caption_embedding_similarity.py --input captions.json --caption_key gemma3:12b_captions --baseline

Cosine is the measure here because we ask the LLM for captions that vary in length,
and cosine ignores length. It only asks whether two captions point the same way, so a
short caption and a long one saying the same thing still score high.

Nothing here reads tiles, so it should work on Mega Man captions too, but that is
untested. Treat the numbers as rough either way. An encoder scores most English fairly
high, and what an LLM gives back for a scene varies a lot, which is what --baseline is
there to show.
"""

import argparse
import itertools
import json
import os
import random
import statistics
import sys
from pathlib import Path

# Only for running this file directly. Imports already find the root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from create_multiple_caption_dataset import make_caption_orderings
from models.sentence_transformers_helper import encode, load_pretrained_encoder

# One per encoder family the helper can load. MiniLM is the default for its size.
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
CLIP = "sentence-transformers/clip-vit-b-32"
T5 = "t5-base"

FLOOR_PAIRS = 20000


def read_caption_sets(path, caption_key, limit=None):
    """The caption list for each scene, plus how many scenes were unusable."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("entries", [])

    sets, skipped = [], 0
    for entry in data:
        if caption_key:
            captions = entry.get(caption_key)
        else:
            # The legacy writer spreads one LLM's captions over caption1..N.
            captions = [entry[key] for key in ["caption", "caption1", "caption2",
                                               "caption3", "caption4"] if key in entry]
        captions = [c.strip() for c in captions or [] if isinstance(c, str) and c.strip()]
        # Nothing to compare with fewer than two
        if len(captions) < 2:
            skipped += 1
            continue
        sets.append(captions)
        if limit and len(sets) >= limit:
            break
    return sets, skipped


def embed(texts, tokenizer, model, batch_size, device):
    """Each distinct string mapped to its unit vector, embedded once."""
    # The ceiling pass hands the same caption back repeatedly, so embed each once
    unique = sorted(set(texts))
    vectors = {}
    for i in range(0, len(unique), batch_size):
        chunk = unique[i:i + batch_size]
        for text, vector in zip(chunk, encode(chunk, tokenizer, model, device)):
            vectors[text] = vector
    return vectors


def pair_sims(captions, vectors):
    """Cosine of every distinct pair."""
    # encode() returns unit vectors, so a dot product is the cosine
    return [float(torch.dot(vectors[a], vectors[b]))
            for a, b in itertools.combinations(captions, 2)]


def summarize(caption_sets, vectors):
    """How alike one scene's captions are, and whether that holds across scenes."""
    means, stds = [], []
    for captions in caption_sets:
        sims = pair_sims(captions, vectors)
        means.append(statistics.fmean(sims))
        # pstdev, since these pairs are the whole set for the scene and not a
        # sample drawn from a larger one
        stds.append(statistics.pstdev(sims) if len(sims) > 1 else 0.0)
    if not means:
        return None
    return {
        "scene_count": len(means),
        "caption_count": sum(len(c) for c in caption_sets),
        "pair_count": sum(len(c) * (len(c) - 1) // 2 for c in caption_sets),
        "average_pairwise_cosine_similarity": statistics.fmean(means),
        "average_scene_standard_deviation": statistics.fmean(stds),
        "standard_deviation_of_scene_averages": statistics.pstdev(means) if len(means) > 1 else 0.0,
    }


def floor_score(caption_sets, vectors):
    """What captions from different scenes score. A floor to read the rest against."""
    flat = [(i, c) for i, captions in enumerate(caption_sets) for c in captions]
    # Sampled rather than every cross-scene pair, which grows with the square of
    # the dataset. Fixed seed so a rerun gives the same floor
    rng = random.Random(0)
    sims = []
    for _ in range(FLOOR_PAIRS):
        (i, a), (j, b) = rng.sample(flat, 2)
        if i != j:            # same scene is the thing we are measuring against
            sims.append(float(torch.dot(vectors[a], vectors[b])))
    if not sims:
        return None
    return {"pair_count": len(sims),
            "average_pairwise_cosine_similarity": statistics.fmean(sims),
            "standard_deviation": statistics.pstdev(sims)}


def ceiling_score(caption_sets, tokenizer, model, batch_size, device):
    """What one caption reworded scores, as a ceiling. One-phrase captions are
    skipped, since there is nothing to reorder."""
    # First caption of each scene only, since reordering the rest measures the
    # same thing again
    usable = [c[0] for c in caption_sets if len(c[0].split(". ")) > 1]
    if not usable:
        return None
    # Same caption count as the real scenes, so the pair counts match
    per_scene = round(statistics.fmean(len(c) for c in caption_sets))
    shuffled = [make_caption_orderings(caption, per_scene) for caption in usable]
    vectors = embed([c for group in shuffled for c in group],
                    tokenizer, model, batch_size, device)
    return summarize(shuffled, vectors)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cosine similarity between an LLM's captions for the same scene.")
    parser.add_argument("--input", required=True,
                        help="JSON dataset from MarioMaker_llm_captions.py.")
    parser.add_argument("--caption_key", default=None,
                        help="Entry key holding the caption list, e.g. "
                             "'gemma3:12b_captions'. Omit for the legacy "
                             "caption/caption1..N fields.")
    parser.add_argument("--encoder", action="append", default=None,
                        help=f"Encoder to embed with, repeated for several. Known good: "
                             f"{MINILM}, {CLIP}, {T5}. Default: {MINILM}.")
    parser.add_argument("--output", default=None,
                        help="Where to write the summary. Prints it otherwise.")
    parser.add_argument("--limit", type=int, default=None, help="Only use this many scenes.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Captions per forward pass. Default: 256.")
    parser.add_argument("--baseline", action="store_true",
                        help="Also score unrelated captions and reorderings of one "
                             "caption, which bracket the number above.")
    parser.add_argument("--device", default=None, help="Torch device. Default: cuda if present.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    caption_sets, skipped = read_caption_sets(args.input, args.caption_key, args.limit)
    if not caption_sets:
        sys.exit(f"ERROR: no usable captions in {args.input} "
                 f"(looked for {args.caption_key or 'caption/caption1..N'})")

    summary = {"input": str(args.input), "caption_key": args.caption_key,
               "device": device, "scene_count": len(caption_sets),
               "skipped_count": skipped, "encoders": {}}

    # One encoder at a time, so only one of them is in memory
    for name in (args.encoder or [MINILM]):
        print(f"Embedding with {name} on {device}...")
        model, tokenizer, _ = load_pretrained_encoder(name, device)
        model.eval()
        vectors = embed([c for captions in caption_sets for c in captions],
                        tokenizer, model, args.batch_size, device)
        score = summarize(caption_sets, vectors)
        if args.baseline:
            score["floor_unrelated_scenes"] = floor_score(caption_sets, vectors)
            score["ceiling_reordered_phrases"] = ceiling_score(
                caption_sets, tokenizer, model, args.batch_size, device)
        summary["encoders"][name] = score
        if score:
            print(f"Average pairwise cosine similarity: "
                  f"{score['average_pairwise_cosine_similarity']:.4f}")

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=4), encoding="utf-8")
        print(f"Wrote caption similarity scores to {args.output}")
    else:
        print(json.dumps(summary, indent=4))
    return summary


if __name__ == "__main__":
    main()
