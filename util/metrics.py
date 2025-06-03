"""
This module provides utility functions for comparing level layouts through various metrics.

The functions in this module operate on level layouts represented as 2D lists/arrays
where each element represents a tile. The specific tile representation can be arbitrary
(characters, integers, etc.) as long as equality comparison is supported between tiles.
"""

from typing import List, Dict, Sequence, TypeVar, Union
import sys
import os
import traceback
from util.sampler import MMNEATSimulator
from captions.caption_match import compare_captions
from util.sampler import scene_to_ascii

# Add the parent directory to the system path to import the extract_tileset function
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'captions'))

from captions.caption_match import TOPIC_KEYWORDS

from create_ascii_captions import assign_caption, extract_tileset
import numpy as np
import json

# Type variable for the tile type
T = TypeVar('T')

#tileset_path = '..\TheVGLC\Super Mario Bros\smb.json'
tileset_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'TheVGLC',
    'Super Mario Bros',
    'smb.json'
)

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

def edit_distance(level1: Sequence[Sequence[T]], level2: Sequence[Sequence[T]]) -> int:
    """
    Calculate the edit distance between two levels, defined as the number of differing tiles.
    
    Args:
        level1: First level layout as a 2D sequence of tiles
        level2: Second level layout as a 2D sequence of tiles
        
    Returns:
        The number of positions where the tiles differ between the two levels
        
    Raises:
        ValueError: If the levels have different dimensions
    """
    if not level1 or not level2:
        raise ValueError("Levels cannot be empty")
        
    if len(level1) != len(level2) or len(level1[0]) != len(level2[0]):
        raise ValueError(
            f"Levels must have same dimensions. Got {len(level1)}x{len(level1[0])} "
            f"vs {len(level2)}x{len(level2[0])}"
        )
    
    return sum(
        tile1 != tile2
        for row1, row2 in zip(level1, level2)
        for tile1, tile2 in zip(row1, row2)
    )

def min_edit_distance(level: Sequence[Sequence[T]], 
                     level_collection: Sequence[Sequence[Sequence[T]]]) -> int:
    """
    Find the minimum edit distance between a level and any level in a collection.
    
    Args:
        level: The level layout to compare against the collection
        level_collection: A sequence of level layouts to compare against
        
    Returns:
        The minimum edit distance found between the input level and any level
        in the collection
        
    Raises:
        ValueError: If level_collection is empty or if any level has different
                   dimensions from the input level
    """
    if not level_collection:
        raise ValueError("Level collection cannot be empty")
        
    try:
        distances = [edit_distance(level, other) for other in level_collection if other != level]
        return min(distances)
    except ValueError as e:
        raise ValueError("All levels in collection must have same dimensions as input level") from e

def average_min_edit_distance(level_collection: List[List[List[int]]]) -> float:
    """
    Calculate the average minimum edit distance between each level and all other levels.
    
    Args:
        level_collection: List of level layouts
        
    Returns:
        Average of minimum edit distances
    """
    if len(level_collection) < 2:
        raise ValueError("Need at least 2 levels to compare")
        
    total_min_distance = 0
    
    for i, level in enumerate(level_collection):
        # Create list of all levels except current one
        other_levels = level_collection[:i] + level_collection[i+1:]
        min_dist = min_edit_distance(level, other_levels)
        total_min_distance += min_dist
        
    return total_min_distance / len(level_collection)

def average_generated_edit_distance(
    generated_levels: List[List[List[int]]],
    game_levels: List[List[List[int]]]
) -> float:
    """
    Calculate the average minimum edit distance between generated levels and game levels

    Args:
        generated_levels (List[List[List[int]]]): Generated level dataset
        game_levels (List[List[List[int]]]): Game level dataset

    Returns:
        float: the average minimum edit distance between generated levels and game levels
    """
    if not generated_levels or not game_levels:
        print("Warning: One or both level lists are empty. Returning 0.0")
        return 0.0
    
    average_distance = 0.0
    
    for level in generated_levels:
        average_distance += min_edit_distance(level, game_levels) # Calculate the min edit distance for each generated level
        
    average_distance /= len(generated_levels)  # Average over all generated levels
    return average_distance
    

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
        caption.replace(f"no broken {feature}s", "").replace(f"no broken {feature}", "")
        for caption in captions
    ]
    return cleaned_captions

def count_broken_feature_mentions(captions: List[str], feature: str) -> float:
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
        return 0.0
    
    # Count mentions of broken feature
    broken_count = sum(
        f"broken {feature}" in caption.lower()
        for caption in cleaned_captions
    )
    
    # Returns percent of broken feature mentions over
    return (broken_count / len(cleaned_captions)) * 100 

def analyze_broken_features_from_data(data: List[Dict], feature: str) -> float:
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
    
    return count_broken_feature_mentions(captions, feature)

def analyze_broken_features_from_scenes(scenes: List[List[List[int]]], feature: str) -> float:
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
    return count_broken_feature_mentions(captions, feature)

# Convenience functions for pipes specifically
def analyze_broken_pipes(data: Union[List[str], List[Dict], List[List[List[int]]]]) -> float:
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
        return count_broken_feature_mentions(data, "pipe")
    elif isinstance(data[0], dict):
        return analyze_broken_features_from_data(data, "pipe")
    else:
        return analyze_broken_features_from_scenes(data, "pipe")

# Convenience functions for cannons specifically
def analyze_broken_cannons(data: Union[List[str], List[Dict], List[List[List[int]]]]) -> float:
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
        return count_broken_feature_mentions(data, "cannon")
    elif isinstance(data[0], dict):
        return analyze_broken_features_from_data(data, "cannon")
    else:
        return analyze_broken_features_from_scenes(data, "cannon")
    
    
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
    false_negatives = 0
    true_negatives = 0
    
    # Normalize the target phrase for comparison
    target_phrase = target_phrase.lower().strip()
        
    for prompt, caption in prompt_caption_pairs:
        # Normalize prompt and caption
        prompt = prompt.lower().strip()
        caption = caption.lower().strip()
        
        # Determine presence of the phrase or topic
        if strict:
            in_prompt = target_phrase in prompt
            in_caption = target_phrase in caption
        else:
            # Non-strict: Check if the target phrase's topic is present
            in_prompt = any(topic in prompt for topic in TOPIC_KEYWORDS if topic in target_phrase)
            in_caption = any(topic in caption for topic in TOPIC_KEYWORDS if topic in target_phrase)
        
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

def percent_perfect_match(prompt_caption_pairs: List[tuple[str, str]]) -> float:
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
        elif exact_matches > 0:
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

def astar_metrics(
    levels: list[dict],  # Each dict should have "scene" and "caption"
    num_runs: int = 3,
    simulator_kwargs: dict = None,
    save_name: str = "astar_metrics_results.json"
) -> list[dict]:
    """
    Runs the SNES A* algorithm on each level multiple times and saves results in the requested format.
    Args:
        levels: List of dicts, each with "scene" (2D int list) and "caption" (str)
        num_runs: Number of runs per level
        simulator_kwargs: kwargs for MMNEATSimulator
        save_name: Filename for output JSON (saved to root directory)
    Returns:
        List of dicts as described in the prompt
    """
    simulator_kwargs = simulator_kwargs or {}
    results = []

    for idx, entry in enumerate(levels):
        scene = entry.get("scene")
        caption = entry.get("caption", None)
        if scene is None:
            continue  # skip if no scene

        ascii_level = scene_to_ascii(scene, id_to_char, True)
        run_metrics = []
        for run in range(num_runs):
            try:
                sim = MMNEATSimulator(ascii_level, **simulator_kwargs)
                output = sim.astar(render=False)
            except Exception as e:
                print(f"Error running A* on level {idx}, run {run}: {e}")
                continue

            metrics = {}
            for line in output.strip().splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    else:
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except Exception:
                            pass
                    metrics[key] = value
            run_metrics.append(metrics)

        # Compute averages
        averages = {}
        if run_metrics:
            keys = set().union(*run_metrics)
            for key in keys:
                values = [m[key] for m in run_metrics if key in m]
                if all(isinstance(v, (int, float)) for v in values):
                    averages[key] = sum(values) / len(values)
                elif all(isinstance(v, bool) for v in values):
                    averages[key] = sum(v for v in values) / len(values)
                else:
                    averages[key] = values

        # Compose result for this scene
        results.append({
            "scene": scene,
            "caption": caption,
            "run_results": run_metrics,
            "averages": averages
        })

    # Save results to root directory
    out_file = os.path.join('.', save_name)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

    return results

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

        # Test average_generated_edit_distance
        print(f"Loaded {len(generated_levels)} generated levels and {len(game_levels)} game levels.")
        print(f"Calculating average min edit distance between generated levels and game levels...")
        avg_edit_distance = average_generated_edit_distance(generated_levels, game_levels)
        print(f"Average Generated Edit Distance: {avg_edit_distance:.2f}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")