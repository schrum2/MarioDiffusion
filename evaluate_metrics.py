import os
import sys
import json

from metrics import (
    average_min_edit_distance,
    count_broken_feature_mentions,
    analyze_phrase_targeting
)
from captions.caption_match import TOPIC_KEYWORDS

def evaluate_metrics(json_dir, output_file):
    """
    Evaluate metrics for JSON level files in a directory and save results to a file.

    Args:
        json_dir (str): Path to the directory containing JSON level files.
        output_file (str): Path to save the evaluation results.
    """
    results = []

    # Iterate through all JSON files in the directory
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_dir, file_name)
            print(f"Processing {file_path}...")

            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract levels and captions
            levels = [entry["scene"] for entry in data if "scene" in entry]
            captions = [entry["caption"] for entry in data if "caption" in entry]

            # Calculate metrics
            metrics = {
                "file_name": file_name,
                "average_min_edit_distance": average_min_edit_distance(levels),
                "broken_pipes_percentage": count_broken_feature_mentions(captions, "pipe"),
                "broken_cannons_percentage": count_broken_feature_mentions(captions, "cannon"),
            }

            # Analyze phrase targeting for all TOPIC_KEYWORDS
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

            # Append metrics to results
            results.append(metrics)

    # Save results to the output file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Metrics saved to {output_file}")
    
    
if __name__ == "__main__":
    if len(sys.argv) < 3: # If fewer than 3 arguments are passed then exit
        print("Usage: python evaluate_metrics.py <model_path> <type>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    eval_type = sys.argv[2]  # "regular" or "absence"

    real_dir = os.path.join(model_path, "samples-from-real-captions")
    random_dir = os.path.join(model_path, "samples-from-random-captions")

    # Build paths for real and random captions
    real_output = os.path.join(model_path, f"evaluation_metrics-{eval_type}-real.json")
    random_output = os.path.join(model_path, f"evaluation_metrics-{eval_type}-random.json")

    
    # Evaluate metrics for both real and random captions
    evaluate_metrics(real_dir, real_output)
    evaluate_metrics(random_dir, random_output)
