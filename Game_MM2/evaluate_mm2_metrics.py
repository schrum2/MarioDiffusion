"""Evaluation metrics for generated Mario Maker 2 levels. The MM2 counterpart of
MarioDiffusion's evaluate_metrics.py. Reports:

  * Broken structure counts per feature (see util/mm2_metrics.py).
  * AMED_self: diversity of the output set.
  * AMED_real (with a real dataset): distance from each level to the training set.
  * Caption adherence when entries carry prompts: score, perfect matches, and
    per-topic phrase targeting. Deterministic prompts only; unconditional samples
    carry no prompts, so adherence is skipped for them automatically.
  * Same-source diversity when entries carry source metadata: max edit distance
    among scenes generated from different captions of one source scene.

Two ways to run:

  * Walk a model directory's sample subdirs (mirrors evaluate_metrics.py) and
    write an evaluation_metrics.json next to each all_levels.json:
        python -m Game_MM2.evaluate_mm2_metrics --model_path MODEL_DIR --game MM2

  * Score a single all_levels.json:
        python -m Game_MM2.evaluate_mm2_metrics --json MODEL-unconditional-samples-short\\all_levels.json --real_json datasets\\MM2_LevelsAndCaptions-regular.json

Run from the repo root (as a module, or imported by a root-level script such as
verify_data_complete.py) so the repo-root packages resolve without a sys.path shim.
"""
import argparse
import json
import os

import util.common_settings as common_settings
from Game_MM2.MarioMaker_create_ascii_captions import build_id_to_char
from captions.MM2_caption_match import compare_captions, extract_topics
from util.mm2_metrics import (
    broken_structure_report,
    average_min_edit_distance,
    average_min_edit_distance_from_real,
    max_edit_distance,
    source_group_key,
)


def phrase_targeting_metrics(prompt_caption_pairs):
    """Precision/recall/F1 per topic across (prompt, generated caption) pairs,
    over the MM2 entity topics the pairs mention rather than a fixed keyword list."""
    extracted = [(extract_topics(p), extract_topics(c)) for p, c in prompt_caption_pairs]
    topics = set()
    for pt, ct in extracted:
        topics |= set(pt) | set(ct)

    stats = {}
    for topic in sorted(topics):
        tp = fp = tn = fn = 0
        for pt, ct in extracted:
            in_prompt = topic in pt
            in_caption = topic in ct
            if in_prompt and in_caption:
                tp += 1
            elif in_caption:
                fp += 1
            elif in_prompt:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        stats[topic] = {
            "true_positives": tp, "false_positives": fp,
            "true_negatives": tn, "false_negatives": fn,
            "precision": precision, "recall": recall, "f1_score": f1,
        }
    return stats


def broken_structure_metrics(scenes, id_to_char):
    report = broken_structure_report(scenes, id_to_char)
    overall = report.pop("_overall")
    num_scenes = overall["num_scenes"]
    for name, entry in report.items():
        entry["broken_percentage_of_feature"] = (
            entry["broken"] / entry["total"] * 100 if entry["total"] else 0.0)
        entry["scenes_with_broken_percentage"] = (
            entry["scenes_with_broken"] / num_scenes * 100 if num_scenes else 0.0)
    overall["scenes_with_any_broken_percentage"] = (
        overall["scenes_with_any_broken"] / num_scenes * 100 if num_scenes else 0.0)
    return {"per_feature": report, "overall": overall}


def source_group_metrics(data, use_gpu=True):
    """Max edit distance within each group of scenes that share a source scene."""
    groups = {}
    for entry in data:
        key = source_group_key(entry)
        if key is None or "scene" not in entry:
            continue
        groups.setdefault(key, []).append(entry["scene"])

    per_group = {}
    for key, scenes in groups.items():
        if len(scenes) < 2:
            continue
        med = max_edit_distance(scenes, use_gpu=use_gpu)
        if med is not None:
            per_group[str(key)] = {"num_scenes": len(scenes), "max_edit_distance": med}

    if not per_group:
        return None
    values = [g["max_edit_distance"] for g in per_group.values()]
    return {
        "num_groups": len(per_group),
        "average_max_edit_distance": sum(values) / len(values),
        "min_max_edit_distance": min(values),
        "max_max_edit_distance": max(values),
        "per_group": per_group,
    }


def compute_metrics(json_path, real_json=None, tileset=common_settings.MM2_TILESET,
                    skip_adherence=False, use_gpu=True, debug=False):
    """Compute the full metrics dict for one all_levels.json. AMED_real is
    included only when `real_json` names an existing file; caption adherence only
    when the entries carry prompt/caption pairs and `skip_adherence` is false."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = [entry["scene"] for entry in data if "scene" in entry]
    if not scenes:
        raise ValueError(f"No scenes found in {json_path}")

    id_to_char = build_id_to_char(tileset)

    metrics = {"file_name": os.path.basename(json_path),
               "total_generated_levels": len(scenes)}

    if debug:
        print(f"Analyzing broken structures in {len(scenes)} scenes...")
    metrics["broken_structures"] = broken_structure_metrics(scenes, id_to_char)

    if debug:
        print("Calculating AMED_self...")
    avg, compared, skipped = average_min_edit_distance(scenes, use_gpu=use_gpu)
    metrics["average_min_edit_distance"] = avg
    metrics["amed_self_levels_compared"] = compared
    metrics["amed_self_levels_skipped"] = skipped

    if real_json and os.path.isfile(real_json):
        if debug:
            print("Calculating AMED_real...")
        with open(real_json, "r", encoding="utf-8") as f:
            real_levels = [entry["scene"] for entry in json.load(f) if "scene" in entry]
        avg, perfect, compared, skipped = average_min_edit_distance_from_real(
            scenes, real_levels, use_gpu=use_gpu)
        metrics["average_min_edit_distance_from_real"] = avg
        metrics["generated_vs_real_perfect_matches"] = perfect
        metrics["percent_perfect_matches"] = perfect / len(scenes) * 100
        metrics["amed_real_levels_compared"] = compared
        metrics["amed_real_levels_skipped"] = skipped
    elif real_json and debug:
        print(f"Skipping AMED_real: real dataset not found at {real_json}")

    pairs = [(entry["prompt"], entry["caption"]) for entry in data
             if entry.get("prompt") and entry.get("caption")]
    if pairs and not skip_adherence:
        if debug:
            print(f"Scoring caption adherence on {len(pairs)} prompt/caption pairs...")
        # Trust scores saved at generation time when present; otherwise rescore.
        scores = [entry["score"] if "score" in entry
                  else compare_captions(entry["prompt"], entry["caption"])
                  for entry in data if entry.get("prompt") and entry.get("caption")]
        perfect = sum(1 for s in scores if s >= 1.0)
        metrics["average_caption_score"] = sum(scores) / len(scores)
        metrics["perfect_caption_matches"] = perfect
        metrics["perfect_caption_match_percentage"] = perfect / len(scores) * 100
        metrics["phrase_targeting"] = phrase_targeting_metrics(pairs)

    group_metrics = source_group_metrics(data, use_gpu=use_gpu)
    if group_metrics:
        if debug:
            print(f"Computed same-source max edit distance over {group_metrics['num_groups']} groups.")
        metrics["same_source_diversity"] = group_metrics

    return metrics


def evaluate_all_levels(json_path, output_file, real_json=None,
                        tileset=common_settings.MM2_TILESET, skip_adherence=False,
                        use_gpu=True, override=False, debug=False):
    """Score one all_levels.json and write the metrics to output_file. Skips when
    output_file already exists unless override is set. Returns the metrics dict,
    or None when skipped or the input could not be read."""
    if os.path.exists(output_file) and not override:
        if debug:
            print(f"Skipping {json_path}: '{output_file}' already exists.")
        return None
    try:
        metrics = compute_metrics(json_path, real_json=real_json, tileset=tileset,
                                  skip_adherence=skip_adherence, use_gpu=use_gpu,
                                  debug=debug)
    except FileNotFoundError:
        print(f"Error: File not found - {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return None
    except ValueError as e:
        print(f"Error evaluating {json_path}: {e}")
        return None

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_file}")
    return metrics


def evaluate_mm2_metrics(model_path, game="MM2", override=False, debug=False, use_gpu=True):
    """Evaluate every all_levels.json under a model directory, mirroring
    evaluate_metrics.py's sample-directory layout, and write an
    evaluation_metrics.json next to each. `game` selects the real dataset
    datasets/{game}_LevelsAndCaptions-regular.json used for AMED_real (skipped
    when that file is absent, so this runs even before the MM2 dataset lands)."""
    if not os.path.exists(model_path):
        print(f"Error: Path does not exist - {model_path}")
        return

    real_json = os.path.join("datasets", f"{game}_LevelsAndCaptions-regular.json")

    lower = model_path.lower()
    fdm = "fdm" in lower
    wgan = "wgan" in lower and "samples" not in lower
    wgan_samples = "wgan" in lower and "samples" in lower
    unconditional = "-unconditional" in lower
    unconditional_short = "unconditional-samples-short" in lower
    unconditional_long = "unconditional-samples-long" in lower
    conditional = "-conditional-" in lower

    if fdm:
        if debug:
            print("Fdm model detected")
        paths = {
            "real": os.path.join(model_path, f"samples-from-real-{game}-captions", "all_levels.json"),
            "real_full": os.path.join(model_path, f"samples-from-real-{game}-captions", "all_levels_full.json"),
            "random": os.path.join(model_path, f"samples-from-random-{game}-captions", "all_levels.json"),
        }
    elif wgan:
        if debug:
            print("Wgan model detected")
        paths = {"short": os.path.join(f"{model_path}-samples", "all_levels.json")}
    elif unconditional:
        if debug:
            print("Unconditional model detected")
        paths = {
            "short": os.path.join(f"{model_path}-unconditional-samples-short", "all_levels.json"),
            "long": os.path.join(f"{model_path}-unconditional-samples-long", "all_levels.json"),
        }
    elif wgan_samples:
        print(f"Skip {model_path} as it is wgan samples")
        return
    elif unconditional_short or unconditional_long:
        print(f"Skip {model_path} as it is an unconditional sample directory")
        return
    elif conditional:
        if debug:
            print("Conditional model detected")
        paths = {
            "real": os.path.join(model_path, f"samples-from-real-{game}-captions", "all_levels.json"),
            "random": os.path.join(model_path, f"samples-from-random-{game}-captions", "all_levels.json"),
            "short": os.path.join(f"{model_path}-unconditional-samples-short", "all_levels.json"),
            "long": os.path.join(f"{model_path}-unconditional-samples-long", "all_levels.json"),
            "real_full": os.path.join(model_path, f"samples-from-real-{game}-captions", "all_levels_full.json"),
        }
    else:
        print(f"Error: Model type not recognized in path - {model_path}")
        raise ValueError(f"Model type not recognized in path: {model_path}")

    for key, json_path in paths.items():
        if not os.path.isfile(json_path):
            if debug:
                print(f"Warning: {key} file not found at {json_path}")
            continue
        suffix = "_full" if key == "real_full" else ""
        output_file = os.path.join(os.path.dirname(json_path), f"evaluation_metrics{suffix}.json")
        if debug:
            print(f"Processing {key} metrics...")
        # Unconditional short/long samples carry no prompts, so caption adherence
        # is skipped automatically inside compute_metrics; AMED_real applies to all.
        evaluate_all_levels(json_path, output_file, real_json=real_json,
                            skip_adherence=False, use_gpu=use_gpu,
                            override=override, debug=debug)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated Mario Maker 2 levels: broken structures, AMED, caption adherence, and same-source diversity.")
    # Directory mode (mirrors evaluate_metrics.py).
    parser.add_argument("--model_path", type=str, default=None, help="Model output directory to walk for all_levels.json files (mirrors evaluate_metrics.py). Mutually exclusive with --json.")
    parser.add_argument("--game", type=str, default="MM2", help="Dataset prefix for AMED_real in --model_path mode: datasets/{game}_LevelsAndCaptions-regular.json")
    # Single-file mode.
    parser.add_argument("--json", type=str, default=None, help="A single generated levels json (all_levels.json). Mutually exclusive with --model_path.")
    parser.add_argument("--real_json", type=str, default=None, help="Training dataset json for AMED_real (single-file mode)")
    parser.add_argument("--tileset", type=str, default=common_settings.MM2_TILESET, help="Tileset json used to encode the scenes")
    parser.add_argument("--output", type=str, default=None, help="Output json (single-file mode; default: evaluation_metrics.json next to --json)")
    parser.add_argument("--skip_adherence", action="store_true", help="Skip prompt-vs-caption scoring (use when the prompts are LLM captions rather than deterministic ones)")
    parser.add_argument("--override", action="store_true", help="Recompute even if the output file already exists")
    parser.add_argument("--cpu", action="store_true", help="Do not use the GPU for edit distance calculations")
    parser.add_argument("--debug", action="store_true", help="Verbose progress output (always on in single-file mode)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if bool(args.model_path) == bool(args.json):
        raise SystemExit("Provide exactly one of --model_path (walk a model dir) or --json (single file).")

    if args.model_path:
        evaluate_mm2_metrics(args.model_path, game=args.game, override=args.override,
                             debug=args.debug, use_gpu=not args.cpu)
    else:
        output = args.output or os.path.join(os.path.dirname(args.json) or ".", "evaluation_metrics.json")
        evaluate_all_levels(args.json, output, real_json=args.real_json, tileset=args.tileset,
                            skip_adherence=args.skip_adherence, use_gpu=not args.cpu,
                            override=args.override, debug=True)
