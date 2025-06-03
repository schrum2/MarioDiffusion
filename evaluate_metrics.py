import os
import json
import argparse
from util.metrics import (
    average_min_edit_distance,
    average_generated_edit_distance,
    count_broken_feature_mentions,
    analyze_phrase_targeting,
    percent_perfect_match,
)
from captions.caption_match import TOPIC_KEYWORDS


def evaluate_all_levels(json_file_path, output_file):
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
        print(f"Successfully loaded data from {json_file_path}")

        levels = [entry["scene"] for entry in data if "scene" in entry]
        captions = [entry["caption"] for entry in data if "caption" in entry]

        print(f"Found {len(levels)} levels and {len(captions)} captions.")

        metrics = {
            "file_name": os.path.basename(json_file_path),
            "average_min_edit_distance": average_min_edit_distance(levels),
            "broken_pipes_percentage": count_broken_feature_mentions(captions, "pipe"),
            "broken_cannons_percentage": count_broken_feature_mentions(captions, "cannon"),
        }
        
        print("Calculating phrase metrics...")

        phrase_metrics = {}
        for keyword in TOPIC_KEYWORDS:
            tp, fp, tn, fn = analyze_phrase_targeting(
                [(caption, caption) for caption in captions], keyword, strict=False
            )
            phrase_metrics[keyword] = {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            }
        metrics["phrase_targeting"] = phrase_metrics

        match_metrics = percent_perfect_match([(caption, caption) for caption in captions])
        metrics["perfect_match_metrics"] = match_metrics

        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: File not found - {json_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def evaluate_metrics(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Path does not exist - {model_path}")
        return

    real_dir = os.path.join(model_path, "samples-from-real-captions")
    random_dir = os.path.join(model_path, "samples-from-random-captions")

    if os.path.isdir(real_dir) and os.path.isdir(random_dir):
        print("Detected Case 1: Subdirectories with `samples-from-real-captions` and `samples-from-random-captions`.")

        real_json_path = os.path.join(real_dir, "all_levels.json")
        real_output_path = os.path.join(real_dir, "evaluation_metrics.json")
        evaluate_all_levels(real_json_path, real_output_path)

        random_json_path = os.path.join(random_dir, "all_levels.json")
        random_output_path = os.path.join(random_dir, "evaluation_metrics.json")
        evaluate_all_levels(random_json_path, random_output_path)

    elif os.path.isfile(os.path.join(model_path, "all_levels.json")):
        print("Detected Case 2: Directory directly containing `all_levels.json`.")

        json_file_path = os.path.join(model_path, "all_levels.json")
        output_file_path = os.path.join(model_path, "evaluation_metrics.json")
        evaluate_all_levels(json_file_path, output_file_path)

    else:
        print(f"Error: Could not find `all_levels.json` in the expected structure under {model_path}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated levels and captions.")
    parser.add_argument("--model_path", type=str, help="Path to the model output directory containing all_levels.json or its subdirectories.")
    # Add more arguments here in the future, for example:
    # parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    # parser.add_argument("--output", type=str, help="Custom output file path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_metrics(args.model_path)
