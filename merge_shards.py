"""
Merge the per-shard checkpoint (.jsonl) files produced by running llm_ascii_to_caption.py with
--shard-index/--shard-count on several machines into a single, ordered captions JSON file.

llm_ascii_to_caption.py names each shard's checkpoint automatically -- never by hand -- by
taking --output and swapping its ".json" for ".shard{index}of{count}.jsonl". For example,
running with --output captions.json and --shard-count 4 produces (across the 4 machines):

    captions.shard0of4.jsonl
    captions.shard1of4.jsonl
    captions.shard2of4.jsonl
    captions.shard3of4.jsonl

Once every machine has finished (or you just want to check progress on a still-running set of
shards, since checkpoints are written incrementally and safe to read at any time), gather those
files into one place (copy them, or point this script at a shared network drive) and run:

    python merge_shards.py --output captions.json --shard-count 4

which reconstructs the same filenames llm_ascii_to_caption.py used and merges them. If you
renamed or moved the files, pass them (or a glob) explicitly instead:

    python merge_shards.py "captions.shard*of4.jsonl" --output captions.json
    python merge_shards.py shard0.jsonl shard1.jsonl --output captions.json

This script only needs the standard library, so it runs fine on a machine that doesn't have
ollama or any of the LLM API packages installed -- e.g. a shared drive server or your laptop.
"""
import argparse
import glob
import json


def default_shard_checkpoint_path(output: str, shard_index: int, shard_count: int) -> str:
    """
    Mirrors llm_ascii_to_caption.py's default_checkpoint_path(): swap --output's ".json" for
    ".jsonl", with a ".shard{index}of{count}" segment inserted before it. Kept as a small,
    self-contained copy here (rather than importing llm_ascii_to_caption.py) so this script
    doesn't pull in ollama/anthropic/openai just to merge some JSON files.
    """
    stem = output[:-5] if output.endswith(".json") else output
    return f"{stem}.shard{shard_index}of{shard_count}.jsonl"


def load_checkpoint_file(path: str) -> dict[int, dict]:
    """Read one checkpoint jsonl file into {global_index: entry}, skipping unparsable lines."""
    entries = {}
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] {path}:{line_num}: skipping unparsable line (likely a crash mid-write)")
                continue
            if "_index" not in entry:
                print(f"[warn] {path}:{line_num}: skipping line with no '_index' field")
                continue
            entries[entry["_index"]] = entry
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="*",
                     help="Shard checkpoint .jsonl file(s) and/or glob patterns (quote globs so your shell "
                          "doesn't expand them, e.g. \"captions.shard*of4.jsonl\"). If omitted, pass "
                          "--shard-count instead and the shard filenames are derived from --output "
                          "the same way llm_ascii_to_caption.py named them.")
    ap.add_argument("--shard-count", type=int, default=None,
                     help="Number of shards to look for, when not passing explicit checkpoint paths/globs. "
                          "Reconstructs each shard's filename from --output, exactly as "
                          "llm_ascii_to_caption.py's --shard-count named it.")
    ap.add_argument("--output", required=True, help="Path to write the merged captions JSON to (and, with "
                                                     "--shard-count, the --output the original shard runs used).")
    args = ap.parse_args()

    if args.checkpoints:
        paths = []
        for pattern in args.checkpoints:
            matches = sorted(glob.glob(pattern))
            paths.extend(matches if matches else [pattern])  # fall back to literal path if glob matched nothing
    elif args.shard_count:
        paths = [default_shard_checkpoint_path(args.output, i, args.shard_count) for i in range(args.shard_count)]
    else:
        ap.error("pass either checkpoint file(s)/glob(s), or --shard-count to derive them from --output")

    merged: dict[int, dict] = {}
    per_file_counts = {}
    for path in paths:
        file_entries = load_checkpoint_file(path)
        per_file_counts[path] = len(file_entries)
        for index, entry in file_entries.items():
            if index in merged:
                # Two shards should never claim the same global index (i % shard_count is a
                # partition), so a collision here means the same checkpoint was included twice,
                # or shard boundaries changed between runs. Keep the first, warn either way.
                print(f"[warn] scene index {index} appears in more than one checkpoint file; keeping the first copy")
                continue
            merged[index] = entry

    ordered = [merged[i] for i in sorted(merged)]
    for entry in ordered:
        entry.pop("_index", None)

    with open(args.output, "w") as f:
        json.dump(ordered, f, indent=2)

    print("\n[merge summary]")
    for path, count in per_file_counts.items():
        print(f"  {path}: {count} scene(s)")
    print(f"  -> merged total: {len(ordered)} scene(s) -> {args.output}")


if __name__ == "__main__":
    main()
