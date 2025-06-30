"""
This module provides utility functions for comparing level layouts through various metrics.

The functions in this module operate on level layouts represented as 2D lists/arrays
where each element represents a tile. The specific tile representation can be arbitrary
(characters, integers, etc.) as long as equality comparison is supported between tiles.
"""
import torch
from typing import List, Dict, Sequence, TypeVar, Union, Tuple
import sys
import os
import numpy as np
import json
from util.sampler import CustomSimulator
from captions.caption_match import compare_captions
from util.sampler import scene_to_ascii
from tqdm import tqdm

# Add the parent directory to the system path to import the extract_tileset function
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'captions'))

from captions.caption_match import TOPIC_KEYWORDS
from create_ascii_captions import assign_caption, extract_tileset


# Type variable for the tile type
T = TypeVar('T')

tileset_path = os.path.join('datasets','smb.json')

# Ensure the tileset path exists
try:
    title_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)
except FileNotFoundError:
    print("\nError: Could not find tileset file!")
    print("\nExpected directory structure:")
    print("GitHub/")
    print("├── MarioDiffusion/")
    print("│   └── util/")
    print("│       └── metrics.py")
    print("└── TheVGLC/")
    print("    └── Super Mario Bros/")
    print("        └── smb.json")

    print("\nActual directory structure:")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Base directory: {base_dir}")
    try:
        for root, dirs, files in os.walk(base_dir):
            level = root.replace(base_dir, '').count(os.sep)
            indent = '│   ' * level
            print(f"{indent}└── {os.path.basename(root)}/")
            for file in files:
                print(f"{indent}    └── {file}")
    except Exception as e:
        print(f"Error walking directory: {e}")

    raise

def edit_distance_tensor(level1: torch.Tensor, level2: torch.Tensor) -> int:
    """Computes edit distance. Here for future use if needed (not used in evaluate metrics)"""
    return (level1 != level2).sum().item()


def average_min_edit_distance(level_collection: List[List[List[int]]], use_gpu=True) -> float:
    """
    Calculate average minimum edit distance (tile-wise) between levels using PyTorch for acceleration.
    """
    if len(level_collection) < 2:
        raise ValueError("Need at least 2 levels to compare")

    levels = torch.tensor(level_collection, dtype=torch.int16)
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    levels = levels.to(device)

    total_min_dist = 0.0
    num_levels = levels.shape[0]

    for i in range(num_levels):
        current = levels[i]  # (H, W)
        others = torch.cat([levels[:i], levels[i+1:]])  # (N-1, H, W)
        diff = (others != current).sum(dim=(1, 2))  # (N-1,)
        total_min_dist += diff.min().item()

    return total_min_dist / num_levels


def average_min_edit_distance_from_real(
    generated_levels: List[List[List[int]]],
    game_levels: List[List[List[int]]],
    use_gpu=True
) -> Tuple[float, int]:
    """
    Calculate average minimum edit distance from generated levels to real levels using GPU.
    """
    if not generated_levels or not game_levels:
        print("Warning: One or both level lists are empty. Returning 0.0, 0.")
        return 0.0, 0
  

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    gen = torch.tensor(generated_levels, dtype=torch.int16).to(device)
    real = torch.tensor(game_levels, dtype=torch.int16).to(device)

    avg_min_dist = 0.0
    perfect_matches = 0

    for level in gen:
        dists = (real != level).sum(dim=(1, 2))
        current_min = dists.min().item()
        if current_min == 0.0:
            perfect_matches += 1
        avg_min_dist += current_min

    return avg_min_dist / len(gen), perfect_matches
    

def remove_absence_captions(captions: List[str], feature: str) -> List[str]:
    """
    Remove captions that only describe the absence of features.
    
    Args:
        captions: List of caption strings
        feature: Feature to check for absence caption(e.g. "pipe" or "cannon")
    Returns:
        List of captions excluding absence descriptions like "no broken pipes"
    """
    # Clean captions by removing "no broken" phrases. Does not remove the caption, rather changes it
    cleaned_captions = [
        caption.replace(f"no broken {feature}s", "").replace(f"no broken {feature}", "").replace(f"no {feature}s", "").replace(f"no {feature}", "").replace(f"no upside down {feature}s", "").replace(f"no upside down {feature}", "").strip()
        for caption in captions
    ]
    return cleaned_captions

def count_broken_feature_mentions(captions: List[str], feature: str, as_percentage_of_feature: bool, as_count: bool) -> float:
    """
    Calculate percentage of captions mentioning a broken feature
    
    Args:
        captions: List of caption strings
        feature: Feature to check ("pipe" or "cannon")
    
    Returns:
        Percentage of captions mentioning broken feature
    """
    # Remove absence captions first
    cleaned_captions = remove_absence_captions(captions, feature)
    if not cleaned_captions:
        print(f"Warning: No captions found after cleaning for feature '{feature}'")
        if as_count: return (0,0)
        return 0.0
    
    # Count mentions of broken feature
    broken_count = sum(
        f"broken {feature}" in caption.lower()
        for caption in cleaned_captions
    )
    
    if as_percentage_of_feature:
        # Count total mentions of the feature
        total_feature_count = sum(
            f"{feature}" in caption.lower()
            for caption in cleaned_captions
        )
        
        if total_feature_count == 0:
            print(f"Warning: No mentions of '{feature}' found in captions")
            if as_count: return (0,0)
            return 0.0
        
        if as_count:
            return broken_count, total_feature_count
        # Return percentage of broken feature mentions over total feature mentions
        return (broken_count / total_feature_count) * 100

    if as_count:
        return broken_count, len(cleaned_captions)    
    # Returns percent of broken feature mentions over total captions
    return (broken_count / len(cleaned_captions)) * 100 

def analyze_broken_features_from_data(data: List[Dict], feature: str, as_instance_of_feature: bool, as_count: bool) -> float:
    """
    Analyze broken features from list of scene/caption dictionaries
    
    Args:
        data: List of dictionaries containing 'caption' keys
        feature: Feature to check ("pipe" or "cannon")
    
    Returns:
        Percentage of scenes with broken feature
    """
    captions = [entry['caption'] for entry in data if 'caption' in entry] # isolate captions
    
    if not captions: # Exception handling for no captions
        print(f"Warning: No captions found in data for feature '{feature}'")
        return 0.0
    
    return count_broken_feature_mentions(captions, feature, as_instance_of_feature, as_count)

def analyze_broken_features_from_scenes(scenes: List[List[List[int]]], feature: str, as_instance_of_feature: bool) -> float:
    """
    Analyze broken features from raw scene data by generating captions
    
    Args:
        scenes: List of scene layouts
        feature: Feature to check ("pipe" or "cannon")
    
    Returns:
        Percentage of scenes with broken feature
    """
    captions = [ # Generate captions for each scene
        assign_caption(
            scene,
            id_to_char,
            char_to_id,
            tile_descriptors,
            describe_locations=False,
            describe_absence=False
        ) 
        for scene in scenes
    ]
    
    if not captions: # Exception handling for no captions
        print(f"Warning: No captions generated for scenes with feature '{feature}'")
        return 0.0
    
    # Use the generated captions to cound broken feature mentions
    return count_broken_feature_mentions(captions, feature, as_instance_of_feature)

# Convenience functions for pipes specifically
def analyze_broken_pipes(data: Union[List[str], List[Dict], List[List[List[int]]]], as_instance_of_feature: bool, as_count: bool) -> float:
    """
    Analyze broken pipes in data, handling different input formats
    
    Args:
        data: Either list of captions, scene/caption dicts, or scenes
        
    Returns:
        Percentage of scenes with broken pipes
    """
    if not data:
        return 0.0
        
    # Determine data type and call appropriate function
    if isinstance(data[0], str):
        return count_broken_feature_mentions(data, "pipe", as_instance_of_feature, as_count)
    elif isinstance(data[0], dict):
        return analyze_broken_features_from_data(data, "pipe", as_instance_of_feature, as_count)
    else:
        return analyze_broken_features_from_scenes(data, "pipe", as_instance_of_feature, as_count)

# Convenience functions for cannons specifically
def analyze_broken_cannons(data: Union[List[str], List[Dict], List[List[List[int]]]], as_instance_of_feature: bool, as_count: bool) -> float:
    """
    Analyze broken cannons in data, handling different input formats
    
    Args:
        data: Either list of captions, scene/caption dicts, or scenes
        
    Returns:
        Percentage of scenes with broken cannons
    """
    if not data:
        return 0.0
        
    # Determine data type and call appropriate function
    if isinstance(data[0], str):
        return count_broken_feature_mentions(data, "cannon", as_instance_of_feature, as_count)
    elif isinstance(data[0], dict):
        return analyze_broken_features_from_data(data, "cannon", as_instance_of_feature, as_count)
    else:
        return analyze_broken_features_from_scenes(data, "cannon", as_instance_of_feature, as_count)
    
    
def analyze_phrase_targeting(
    prompt_caption_pairs: List[tuple[str, str]],
    target_phrase: str,
    strict: bool
) -> tuple[int, int, int, int]:
    """
    Analyze how well the model targets specific phrases in generation    
    
    Args:
        prompt_caption_pairs: List of (input_prompt, generated_caption) pairs
        target_phrase: Specific phrase to look for (e.g. "two pipes")
        
    Returns:
        Tuple containing:
        - true_positives: Count where phrase apprears in both prompt and generation
        - false_positives: Count where phrase appears in generation but not prompt
        - false_negatives: Count where phrase appears in prompt but not generation
        - true_negatives: Count where phrase does not appear in either
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    # Normalize the target phrase for comparison
    target_phrase = target_phrase.lower().strip()
    
    # Extract relevant keywords from the target phrase
    relevant_keywords = [kw for kw in TOPIC_KEYWORDS if kw in target_phrase]
        
    for prompt, caption in prompt_caption_pairs:
        # Normalize prompt and caption
        prompt = prompt.lower().strip()
        caption = caption.lower().strip()
        
        # Determine presence of the phrase or topic
        if strict:
            in_prompt = target_phrase in prompt
            in_caption = target_phrase in caption
        else:
            # Look for any relevant keyword from the target phrase
            in_prompt = any(kw in prompt for kw in relevant_keywords)
            in_caption = any(kw in caption for kw in relevant_keywords)
        
        # Update counts based on presence
        if in_prompt and in_caption:
            true_positives += 1
        elif not in_prompt and in_caption:
            false_positives += 1
        elif not in_prompt and not in_caption:
            true_negatives += 1
        else:  # in_prompt and not in_caption
            false_negatives += 1
    
    return (true_positives, false_positives, true_negatives, false_negatives)

def calculate_phrase_metrics(
    prompt_caption_pairs: List[tuple[str, str]],
    target_phrase: str,
    strict: bool
) -> dict:
    """
    Calculate precision, recall, and F1 score for a specific phrase.
    
    Args:
        prompt_caption_pairs: List of (input_prompt, generated_caption) pairs
        target_phrase: Specific phrase to analyze (e.g., "two pipes")
    
    Returns:
        A dictionary containing:
        - true_positives
        - false_positives
        - true_negatives
        - false_negatives
        - precision
        - recall
        - f1_score
    """
    # Get counts from analyze_phrase_targeting
    tp, fp, tn, fn = analyze_phrase_targeting(prompt_caption_pairs, target_phrase, strict)
    
    # Calculate metrics
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total": total
    }

def percent_perfect_match(prompt_caption_pairs: List[tuple[str, str]]) -> Dict[str, Union[int, float]]:
    """
    Calculate the percentage of perfect matches between prompts and captions.
    
    Args:
        prompt_caption_pairs: List of (input_prompt, generated_caption) pairs
    
    Returns:
        Percentage of perfect matches
    """
    if not prompt_caption_pairs:
        raise ValueError("The list of prompt-caption pairs cannot be empty")
    
    total_pairs = len(prompt_caption_pairs)
    perfect_match_count = 0
    partial_match_count = 0
    no_match_count = 0
    
    for prompt, caption in prompt_caption_pairs:
        if not isinstance(prompt, str) or not isinstance(caption, str):
            raise ValueError("Both prompt and caption must be strings")
        compare_score, exact_matches, partial_matches, excess_phrases = compare_captions(
            prompt, caption, return_matches=True
        )

        # Check for perfect match (all phrases match exactly)
        if compare_score == 1.0 and not excess_phrases:
            perfect_match_count += 1
        elif len(exact_matches) > 0:
            # Check for at least one matching phrase
            partial_match_count += 1
        else:
            # No matches at all
            no_match_count += 1
            
    # Calculate percentages
    perfect_match_percentage = (perfect_match_count / total_pairs) * 100
    partial_match_percentage = (partial_match_count / total_pairs) * 100
    no_match_percentage = (no_match_count / total_pairs) * 100
    
    return {
        "perfect_match_percentage": perfect_match_percentage,
        "perfect_match_count": perfect_match_count,
        "partial_match_percentage": partial_match_percentage,
        "partial_match_count": partial_match_count,
        "no_match_percentage": no_match_percentage,
        "no_match_count": no_match_count
    }

def astar_metrics(
    levels: list[dict]|list[list[str]],  # Each dict should have "scene" and "caption", each list of lists should be in ascii format
    num_runs: int = 1,
    simulator_kwargs: dict = None,
    output_json_path: str = None,
    save_name: str = "astar_result.jsonl"
) -> tuple[List[dict], dict]:
    """
    Runs the A* algorithm on each level multiple times and saves results in JSONL format.
    Args:
        levels: List of dicts (one-hot encoded json data) or list of lists of strings (raw ascii chars)
        num_runs: Number of runs per level
        simulator_kwargs: kwargs for CustomSimulator
        output_json_path: Path to input JSON file (saves in root directory if None)
        save_name: Filename for output JSONL
    Returns:
        A tuple: A list of dicts as results and a dict of overall averages.
    """
    simulator_kwargs = simulator_kwargs or {}
    results = []
    per_level_averages = []

    # Determine output directory
    if output_json_path is not None:
        output_dir = os.path.dirname(output_json_path)
    else:
        output_dir = '.'
    out_file = os.path.join(output_dir, save_name)

    if not levels:
        raise RuntimeError("No levels provided to astar_metrics. Exiting.")

    # Open the file ONCE in write mode to clear and write all results
    with open(out_file, "w") as f:
        for idx, entry in enumerate(tqdm(levels, desc="A* metrics", unit="level")):
            # Most calls should be a dict, but we should allow for calling this function with just an ascii level, so we add support here
            if isinstance(entry, dict):
                scene = entry.get("scene")
                caption = entry.get("caption", None)

                ascii_level = scene_to_ascii(scene, id_to_char, True)
            else:
                scene=entry
                ascii_level = entry
                caption="This level has no caption"
            

            run_metrics = []
            for run in range(num_runs):
                try:
                    sim = CustomSimulator(ascii_level, **simulator_kwargs)
                    output = sim.astar(render=False)
                    # # Enable rendering if needed (for debugging)
                    # output = sim.astar()
                except Exception as e:
                    if not run_metrics:
                        raise RuntimeError(
                            f"No output to parse for level {idx} (caption: {caption}). Exiting as requested."
                        )

                metrics = {}
                for line in output.strip().splitlines():
                    if ':' not in line:
                        # If a line does not contain a colon, it might be an error message
                        if "Invalid or corrupt jarfile" in line or "Exception" in line or "Error" in line:
                            print(
                                f"Warning: Error from A* agent for level {idx}, run {run} (caption: {caption}): {line}"
                            )
                            # Continue to the next run
                            continue
                    elif ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        if value.lower() in ("true", "false"):
                            value = value.lower() == "true"
                        else:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        metrics[key] = value
                        
                if metrics:
                    run_metrics.append(metrics)
                else:
                    print(
                        f"Warning: No metrics parsed for level {idx}, run {run} (caption: {caption}). Skipping this run."
                    )
                    continue

            # Compute averages and medians
            averages = {}
            medians = {}
            standard_deviations = {}
            if run_metrics:
                keys = set().union(*run_metrics)
                for key in keys:
                    values = [m[key] for m in run_metrics if key in m]
                    if not values:
                        raise RuntimeError(
                            f"No valid metrics found for key '{key}' in level {idx} (caption: {caption}). Exiting as requested."
                        )
                    if all(isinstance(v, (int, float)) for v in values):
                        averages[key] = sum(values) / len(values)
                        medians[key] = np.median(values)
                        standard_deviations[key] = np.std(values)
                    elif all(isinstance(v, bool) for v in values):
                        averages[key] = sum(v for v in values) / len(values)
                        medians[key] = np.median([int(v) for v in values])
                        standard_deviations[key] = np.std([int(v) for v in values])

            # Compose result for this scene
            result = {
                "scene": scene,
                "caption": caption,
                "run_results": run_metrics,
                "averages": averages,
                "medians": medians,
                "standard_deviations": standard_deviations
            }
            results.append(result)
            per_level_averages.append(averages)

            # Write this result as a JSON line
            f.write(json.dumps(result) + "\n")

            # Uncomment below to flush after each write (not recommended for performance)
            #f.flush()  # Ensure each result is written immediately
            #os.fsync(f.fileno())  # Ensure file is flushed to disk

    print(f"Results saved to {out_file}")

    # Compute the overall averages of all averages computed across all runs on all scenes
    overall_averages = {}
    if per_level_averages:
        all_keys = set().union(*per_level_averages)
        for key in all_keys:
            values = [avg[key] for avg in per_level_averages if key in avg]
            if values:
                overall_averages[key] = sum(values) / len(values)
        # Save to a separate JSON file
        summary_file = os.path.splitext(save_name)[0] + "_overall_averages.json"
        summary_path = os.path.join(output_dir, summary_file)
        with open(summary_path, "w") as f:
            json.dump(overall_averages, f, indent=2)
        print(f"Overall averages saved to {summary_path}")

    return results, overall_averages if results else None

if __name__ == "__main__":
    # Base directory for datasets
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    """
    Expected Results (based on previous runs):
    
    Super Mario Bros 1:
    - Average Edit Distance: ~10.1
    
    Super Mario Bros 2:
    - Average Edit Distance: ~11.3
    
    Super Mario Land:
    - Average Edit Distance: ~14.6

    Combined SMB1+2:
    - Average Edit Distance: ~10.6
    
    All Mario Games:
    - Average Edit Distance: ~11.6
    """
    # Paths to the JSON files
    generated_file_path = "c:\\Users\\salas2\\Documents\\GitHub\\MarioDiffusion\\TESTING_Broken_Features.json"
    game_levels_file_path = "c:\\Users\\salas2\\Documents\\GitHub\\MarioDiffusion\\datasets\\SMB1_LevelsAndCaptions-regular.json"

    try:
        # Load the generated dataset
        with open(generated_file_path, "r") as generated_file:
            generated_data = json.load(generated_file)
            generated_levels = [entry["scene"] for entry in generated_data if "scene" in entry]

        # Load the actual game levels dataset
        with open(game_levels_file_path, "r") as game_levels_file:
            game_data = json.load(game_levels_file)
            game_levels = [entry["scene"] for entry in game_data if "scene" in entry]

        # Test average_min_edit_distance_from_real
        print(f"Loaded {len(generated_levels)} generated levels and {len(game_levels)} game levels.")
        print(f"Calculating average min edit distance between generated levels and game levels...")
        avg_edit_distance = average_min_edit_distance_from_real(generated_levels, game_levels)
        print(f"Average Generated Edit Distance: {avg_edit_distance:.2f}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")