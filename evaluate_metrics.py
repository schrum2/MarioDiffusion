import os
import json
import argparse
from util.metrics import (
    average_min_edit_distance,
    average_min_edit_distance_from_real,
    count_broken_feature_mentions,
    analyze_phrase_targeting,
    percent_perfect_match,
    calculate_phrase_metrics,
)
from captions.caption_match import TOPIC_KEYWORDS


def evaluate_all_levels(json_file_path, output_file, original_dataset, random_test):
    """
    Evaluate metrics for a single `all_levels.json` file and save results.

    Args:
        json_file_path (str): Path to the `all_levels.json` file.
        output_file (str): Path to save the evaluation results.
        original_dataset (str): Path to the original dataset for comparison.
    """
    try:
        # Load the generated dataset
        with open(json_file_path, "r") as f:
            data = json.load(f)
        print(f"Successfully loaded data from {json_file_path}")

        levels = [entry["scene"] for entry in data if "scene" in entry]
        captions = [entry["caption"] for entry in data if "caption" in entry]

        print(f"Found {len(levels)} generated levels and {len(captions)} captions.")

        metrics = {
            "file_name": os.path.basename(json_file_path),
            "average_min_edit_distance": average_min_edit_distance(levels),
            "broken_pipes_percentage_in_dataset": count_broken_feature_mentions(captions, "pipe", as_percentage_of_feature=False),
            "broken_pipes_percentage_of_pipes": count_broken_feature_mentions(captions, "pipe", as_percentage_of_feature=True),
            "broken_cannons_percentage_in_dataset": count_broken_feature_mentions(captions, "cannon", as_percentage_of_feature=False),
            "broken_cannons_percentage_of_cannons":count_broken_feature_mentions(captions, "cannon", as_percentage_of_feature=True),
        }
        
        # Only want samples-from-real-captions to process the original dataset and make 'prompts
        # Only want samples-from-random-captions to process from the RandomTest dataset and make prompts

        if original_dataset is not None: # change this to determine if the key == samples from real captions
            # With the original dataset, calculate average_min_edit_distance_from_real
            with open(original_dataset, "r") as original_file:
                original_data = json.load(original_file)
                original_levels = [entry["scene"] for entry in original_data if "scene" in entry]
                prompts = [entry["caption"] for entry in original_data if "caption" in entry]
            print(f"Found {len(original_levels)} original levels and captions.")
          
            metrics["average_min_edit_distance_from_real"] = average_min_edit_distance_from_real(levels, original_levels)
        
        elif random_test is not None: # change this to run if key == samples from random captions
            with open(random_test, "r")as random_test:
                random_data = json.load(random_test)
                random_levels = [entry["scene"] for entry in random_data if "scene" in entry]
                prompts = [entry["caption"] for entry in random_data if "caption" in entry]
            print(f"Found {len(random_levels)} random levels and captions.")
            metrics["average_min_edit_distance_from_real"] = average_min_edit_distance_from_real(levels, random_levels)    
        
        # If prompts was created, meaning that we can do analysis between prompts and captions
        if prompts is not None:
            print("Calculating phrase metrics...")

            phrase_metrics = {}
            for keyword in TOPIC_KEYWORDS:
                tp, fp, tn, fn = analyze_phrase_targeting(
                    list(zip(prompts, captions)), keyword, strict=True
                )
                phrase_metrics[keyword] = {
                    "true_positives": tp,
                    "false_positives": fp,
                    "true_negatives": tn,
                    "false_negatives": fn,
                }
            metrics["phrase_targeting"] = phrase_metrics

            match_metrics = percent_perfect_match(list(zip(prompts, captions)))
            print(f"Perfect match metrics complete!")

            metrics["perfect_match_metrics"] = match_metrics
        else: 
            print("Phrase targeting is not performed for {json_file_path} as it is not generated with prompts.")
        
        # # adding phrase metrics
        # phrase_metrics = calculate_phrase_metrics(list(zip(prompts, captions)), target_phrase="pipe",strict=True)
        # metrics["calculate_phrase_metrics"] = phrase_metrics

        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: File not found - {json_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def evaluate_metrics(model_path, original_dataset, random_test):
    """
    Evaluate metrics for the given model path.

    Args:
        model_path (str): Path to the model directory.
        original_dataset (str): Path to the original dataset for comparison (optional).
    """
    if not os.path.exists(model_path):
        print(f"Error: Path does not exist - {model_path}")
        return
    
    # Define paths for the four expected all_levels.json files
    paths = {
        "real": os.path.join(model_path, "samples-from-real-captions", "all_levels.json"), # Location of these directories will change
        "random": os.path.join(model_path, "samples-from-random-captions", "all_levels.json"),
        "short": os.path.join(f"{model_path}-unconditional-samples-short", "all_levels.json"),
        "long": os.path.join(f"{model_path}-unconditional-samples-long", "all_levels.json"),
    }

    for key, json_path in paths.items():
        if os.path.isfile(json_path):
            output_file = os.path.join(os.path.dirname(json_path), "evaluation_metrics.json")
            print(f"Processing {key} metrics...")
            
            # These make use of the original dataset
            if key == "real" or key == "short":
                evaluate_all_levels(json_path, output_file, original_dataset, None)
            elif key == "random":
                evaluate_all_levels(json_path, output_file, None, random_test)
            elif key == "long":
                evaluate_all_levels(json_path, output_file, None, None)
        else:
            print(f"Warning: {key} file not found at {json_path}")
        


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated levels and captions.")
    parser.add_argument("--model_path", type=str, help="Path to the model output directory containing all_levels.json or its subdirectories.")
    parser.add_argument("--original_dataset", type=str, default=None, help="Path to the original dataset for comparison (optional).")
    #parser.add_argument("--game", type=str, default="Mario", choices=["Mario", "LR"], help="Game type for which to evaluate levels.")
    parser.add_argument("--random_test", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_metrics(args.model_path, args.original_dataset, args.random_test)
