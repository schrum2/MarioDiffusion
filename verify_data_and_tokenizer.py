"""Verify that a tokenizer recognizes every token present across all captions in a dataset.

This is a sanity-check tool for caption datasets: it loads a JSON (or JSONL) dataset and a
saved tokenizer (.pkl), tokenizes every caption, and reports any token that is *not* in the
tokenizer's vocabulary. Such "illegal" tokens would be silently skipped by Tokenizer.encode()
at train/inference time, so catching them up front prevents quietly mis-encoded captions.

Although it lives in MarioDiffusion, it is equally useful in the MariOver repo for validating
that LLM-generated captions never introduce tokens the tokenizer has never seen.

Examples:
    # Standard MarioDiffusion dataset + tokenizer
    python verify_data_and_tokenizer.py \
        --json Game_Mario/DATA/Mar1and2_LevelsAndCaptions-regular.json \
        --pkl Game_Mario/DATA/Mar1and2_Tokenizer-regular.pkl

Exit codes: 0 = all tokens recognized, 1 = illegal tokens found, 2 = could not run (bad input).
"""

import os
import sys
import json
import argparse
from collections import Counter, defaultdict

from tokenizer import Tokenizer


def load_dataset(path):
    """Load a dataset from a .json (list) or .jsonl file.

    Returns a list of entries. Each entry is typically a dict (with a 'caption' field),
    but a list of plain caption strings is also supported.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    if path.endswith(".jsonl"):
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}")
        return entries

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at the top level of {path}, got {type(data).__name__}")
    return data


def extract_captions(entries):
    """Yield (entry_index, caption_text) for every entry with a usable 'caption' string.

    Both dataset shapes we care about store the text under 'caption':
    RandomTest entries are `{"caption": ...}` and LevelsAndCaptions entries are
    `{"prompt", "scene", "caption"}`. Entries that lack a string caption (missing key,
    null, or wrong type) are skipped and counted so the caller can warn about them --
    handy for catching malformed LLM-generated captions.
    """
    results = []
    skipped = 0

    for idx, entry in enumerate(entries):
        caption = entry.get("caption") if isinstance(entry, dict) else None
        if isinstance(caption, str):
            results.append((idx, caption))
        else:
            skipped += 1

    return results, skipped


def verify(captions, tokenizer):
    """Tokenize every caption and find tokens absent from the tokenizer vocabulary.

    Returns:
        illegal_counts: Counter mapping unknown token -> number of occurrences
        illegal_examples: dict mapping unknown token -> list of (entry_index, caption)
        total_tokens: total number of tokens seen across all captions
        unique_tokens: set of all distinct tokens seen
    """
    vocab = tokenizer.token_to_id
    illegal_counts = Counter()
    illegal_examples = defaultdict(list)
    total_tokens = 0
    unique_tokens = set()

    for idx, caption in captions:
        for tok in tokenizer.tokenize(caption):
            total_tokens += 1
            unique_tokens.add(tok)
            if tok not in vocab:
                illegal_counts[tok] += 1
                # Record each offending entry once (captions are processed in index
                # order, so a token repeated within one caption stays consecutive).
                if not illegal_examples[tok] or illegal_examples[tok][-1][0] != idx:
                    illegal_examples[tok].append((idx, caption))

    return illegal_counts, illegal_examples, total_tokens, unique_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Verify that a tokenizer recognizes every token across all captions in a dataset."
    )
    parser.add_argument("--json", required=True,
                        help="Path to the dataset (.json list or .jsonl) containing captions.")
    parser.add_argument("--pkl", required=True,
                        help="Path to the saved tokenizer (.pkl) to validate against.")
    args = parser.parse_args()

    # --- Load tokenizer ---
    if not os.path.exists(args.pkl):
        print(f"ERROR: Tokenizer file does not exist: {args.pkl}", file=sys.stderr)
        return 2
    tokenizer = Tokenizer()
    try:
        tokenizer.load(args.pkl)
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer from {args.pkl}: {e}", file=sys.stderr)
        return 2

    # --- Load dataset ---
    try:
        entries = load_dataset(args.json)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    captions, skipped = extract_captions(entries)

    print(f"Dataset:   {args.json}  ({len(entries)} entries)")
    print(f"Tokenizer: {args.pkl}  (vocab size {tokenizer.get_vocab_size()})")
    print(f"Captions found: {len(captions)}")
    if skipped:
        print(f"  Note: {skipped} entr{'y' if skipped == 1 else 'ies'} skipped (no string 'caption' field).")

    if not captions:
        print("\nERROR: No captions were found to verify. Check --json.", file=sys.stderr)
        return 2

    illegal_counts, illegal_examples, total_tokens, unique_tokens = verify(captions, tokenizer)

    print(f"\nTotal tokens scanned: {total_tokens}")
    print(f"Distinct tokens:      {len(unique_tokens)}")

    if not illegal_counts:
        print("\nSUCCESS: Every token across all captions is recognized by the tokenizer.")
        return 0

    num_illegal = len(illegal_counts)
    print(f"\nFAILURE: Found {num_illegal} illegal token(s) not present in the tokenizer vocabulary:")
    # Sort by frequency (descending), then alphabetically for stable output.
    for tok, count in sorted(illegal_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"\n  {tok!r}  (appears {count} time{'s' if count != 1 else ''})")
        for idx, caption in illegal_examples[tok]:
            print(f"      entry {idx}: {caption}")

    print(f"\nVerification failed: {num_illegal} illegal token(s) found across {sum(illegal_counts.values())} occurrence(s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
