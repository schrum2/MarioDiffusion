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

def real_data():
    real_levels_file_path = "datasets\\Mar1and2_LevelsAndCaptions-regular.json"
    # Open full dataset
    with open(real_levels_file_path, "r") as game_levels_file:
        game_data = json.load(game_levels_file)
        game_levels = [entry["scene"] for entry in game_data if "scene" in entry]
        
    avg_edit_distance_full = average_min_edit_distance(game_levels)
    
    # Resample levels to 100 samples
    if len(game_levels) > 100:
        increment = len(game_levels) // (100 + 1)
        reduced_data = [game_levels[(i + 1) * increment] for i in range(100)]
    
        if len(reduced_data) != 100:
            raise RuntimeError(f"Sample limit mismatched: Expected 100 samples, got {len(reduced_data)} after sampling.")
    # calculate avg min edit distance self on this
    avg_edit_distance_100 = average_min_edit_distance(reduced_data)

    results = {
        "average_min_edit_distance_full": avg_edit_distance_full,
        "average_min_edit_distance_100": avg_edit_distance_100
    }
    # Save to a json file in a folder above datasets called "real_data"
    datasets_dir = os.path.dirname(real_levels_file_path)
    parent_dir = os.path.dirname(datasets_dir)
    output_dir = os.path.join(parent_dir, "real_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "real_data_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

def evaluate_all_levels(json_file_path, output_file, game, key, debug):
    """
    Evaluate metrics for a single `all_levels.json` file and save results.

    Args:
        json_file_path (str): Path to the `all_levels.json` file.
        output_file (str): Path to save the evaluation results.
        game (str): Prefix for accessing correct datasets
        key (str): Indicates the case we are handling
    """
    try:
        # Check if evaluation_metrics.json already exists in the same directory
        if os.path.exists(output_file):
            if debug: print(f"Skipping {key}: '{output_file}' already exists.")
            return
        
        # Load the generated dataset from the path
        with open(json_file_path, "r") as f:
            data = json.load(f)
        if debug: print(f"Successfully loaded data from {json_file_path}")

        # Strip the prompts, scenes, and generated captions from the current all_levels.json
        prompts = [entry["prompt"] for entry in data if "prompt" in entry]
        levels = [entry["scene"] for entry in data if "scene" in entry]
        captions = [entry["caption"] for entry in data if "caption" in entry]

        #print(f"Found {len(prompts)} prompts, {len(levels)} generated levels, and {len(captions)} generated captions.")

        broken_pipe_count, total_scenes = count_broken_feature_mentions(captions, "pipe", as_percentage_of_feature=False, as_count=True)
        broken_pipes_percentage_in_dataset = (broken_pipe_count / total_scenes) * 100

        broken_pipe_count, pipe_scenes = count_broken_feature_mentions(captions, "pipe", as_percentage_of_feature=True, as_count=True)
        broken_pipes_percentage_of_pipes = (broken_pipe_count / pipe_scenes) * 100

        broken_cannon_count, total_scenes = count_broken_feature_mentions(captions, "cannon", as_percentage_of_feature=False, as_count=True)
        broken_cannons_percentage_in_dataset = (broken_cannon_count / total_scenes) * 100

        broken_cannon_count, cannon_scenes = count_broken_feature_mentions(captions, "cannon", as_percentage_of_feature=True, as_count=True)
        broken_cannons_percentage_of_cannons = (broken_cannon_count / cannon_scenes) * 100

        metrics = {
            "file_name": os.path.basename(json_file_path),
            "average_min_edit_distance": average_min_edit_distance(levels),
            "broken_pipes_percentage_in_dataset": broken_pipes_percentage_in_dataset,
            "broken_pipes_percentage_of_pipes": broken_pipes_percentage_of_pipes,
            "broken_cannons_percentage_in_dataset": broken_cannons_percentage_in_dataset,
            "broken_cannons_percentage_of_cannons": broken_cannons_percentage_of_cannons,

            "total_generated_levels": total_scenes,
            "broken_pipes_count": broken_pipe_count,
            "broken_cannons_count": broken_cannon_count,
            "total_pipes": pipe_scenes,
            "total_cannons": cannon_scenes
        }
        
        if key == "real" or key == "short" or key == "random" or key == "real_full": 
            # With the original dataset, calculate average_min_edit_distance_from_real
            original_dataset_path = os.path.join("datasets", f"{game}_LevelsAndCaptions-regular.json")
            with open(original_dataset_path, "r") as original_file:
                original_data = json.load(original_file)
                original_levels = [entry["scene"] for entry in original_data if "scene" in entry]
            
            metrics["average_min_edit_distance_from_real"], metrics["generated_vs_real_perfect_matches"] = average_min_edit_distance_from_real(levels, original_levels)
            metrics["percent_perfect_matches"] = metrics["generated_vs_real_perfect_matches"] / len(levels) * 100 # What % of the generated dataset are real levels
 
        # If prompts was created, meaning that we can do analysis between prompts and captions
        if prompts is not None and (key != "short" and key != "long"):
            if debug: print("Calculating phrase metrics...")

            phrase_metrics = {}
            for keyword in TOPIC_KEYWORDS:
                metrics_dict = calculate_phrase_metrics(
                    list(zip(prompts, captions)), keyword, strict=True
                )
                phrase_metrics[keyword] = metrics_dict
                
            metrics["phrase_targeting"] = phrase_metrics
            
            match_metrics = percent_perfect_match(list(zip(prompts, captions)))

            metrics["perfect_match_metrics"] = match_metrics
        
        else: 
            if debug: print(f"Phrase targeting is not performed for {json_file_path} as it is not generated with prompts.")
        
        
        # Add resulting metrics to a JSON file in the same directory as all_levels.json
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: File not found - {json_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def evaluate_metrics(model_path, game, override, debug=False):
    """
    Evaluate metrics for the given model path.

    Args:
        model_path (str): Path to the model directory.
        game (str): Game prefix name for accessing datasets
    """
    # Determine the model type from naming convention
    fdm = "fdm" in model_path.lower()
    wgan = "wgan" in model_path.lower() and "samples" in model_path.lower()
    unconditional_short = "unconditional-samples-short" in model_path.lower()
    unconditional_long = "unconditional-samples-long" in model_path.lower()
    conditional = "-conditional-" in model_path.lower()
    
    if not os.path.exists(model_path):
        print(f"Error: Path does not exist - {model_path}")
        return
    
    if fdm: 
        if debug: print("Fdm model detected")
        paths = {
            "real": os.path.join(model_path, f"samples-from-real-{game}-captions", "all_levels.json"), # Location of these directories will change
            "random": os.path.join(model_path, f"samples-from-random-{game}-captions", "all_levels.json")
        }
    elif wgan:
        if debug: print("Wgan model detected")
        paths = {
            "short": os.path.join(model_path, "all_levels.json")
        }
    elif unconditional_short:
        if debug: print("Unconditional model (short samples) detected")
        paths = {
            "short": os.path.join(model_path, "all_levels.json")
        }
    elif unconditional_long:
        if debug: print("Unconditional model (long samples) detected")
        paths = {
            "long": os.path.join(model_path, "all_levels.json")
        }
    elif conditional: # Define paths for the four expected all_levels.json files for a conditional model
        if debug: print("Conditional model detected")
        paths = {
            "real": os.path.join(model_path, f"samples-from-real-{game}-captions", "all_levels.json"), # Location of these directories will change
            "random": os.path.join(model_path, f"samples-from-random-{game}-captions", "all_levels.json"),
            "short": os.path.join(f"{model_path}-unconditional-samples-short", "all_levels.json"),
            "long": os.path.join(f"{model_path}-unconditional-samples-long", "all_levels.json"),
            "real_full": os.path.join(model_path, f"samples-from-real-{game}-captions", "all_levels_full.json"),
        }

    for key, json_path in paths.items():
        if os.path.isfile(json_path):
            
            if key == "real_full":
                output_file = os.path.join(os.path.dirname(json_path), "evaluation_metrics_full.json")
            else:
                output_file = os.path.join(os.path.dirname(json_path), "evaluation_metrics.json")
            
            if override and os.path.exists(output_file):
                print(f"Override enabled: deleting existing {output_file}")
                os.remove(output_file)

            if debug: print(f"Processing {key} metrics...")
            evaluate_all_levels(json_path, output_file, game, key, debug)
        else:
            if debug: print(f"Warning: {key} file not found at {json_path}")
        


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated levels and captions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model output directory containing all_levels.json or its subdirectories.")
    parser.add_argument("--game", type=str, default="Mar1and2", help="Game prefix for which to evaluate levels.")
    parser.add_argument("--override", action="store_true", help="Override all previously existing evaluation_metrics.json files and re-run calculations")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--real_data", action="store_true", help="Will create real data for avg min edit distance self when set")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.real_data:
        real_data()
    else:
        evaluate_metrics(args.model_path, args.game, args.override, args.debug)
    