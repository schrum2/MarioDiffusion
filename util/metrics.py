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

print(f"Looking for tileset at: {os.path.abspath(tileset_path)}")  # Debug print

try:
    title_chars, id_to_char, char_to_id, title_descriptors = extract_tileset(tileset_path)
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

# DELETE
def test_edit_distances():
    """Test the edit distance functions on SMB1 levels"""
    
    # Load some test levels
    with open("SMB1_LevelsAndCaptions-regular.json", 'r') as f:
        data = json.load(f)
        levels = [entry['scene'] for entry in data if 'scene' in entry][:20]  # Take first 5 levels
        
    print("\nTesting edit_distance:")
    print("-" * 30)
    # Test edit_distance between first two levels
    try:
        dist = edit_distance(levels[0], levels[16])
        print(f"Edit distance between level 0 and 1: {dist}")
    except Exception as e:
        print(f"Error computing edit_distance: {e}")
        
    print("\nTesting min_edit_distance:")
    print("-" * 30)
    # Test min_edit_distance for first level against others
    try:
        min_dist = min_edit_distance(levels[0], levels[1:])
        print(f"Min edit distance for level 0: {min_dist}")
    except Exception as e:
        print(f"Error computing min_edit_distance: {e}")
        
    print("\nTesting average_min_edit_distance:")
    print("-" * 30)
    # Test average minimum edit distance across all levels
    try:
        avg_dist = average_min_edit_distance(levels)
        print(f"Average minimum edit distance: {avg_dist}")
    except Exception as e:
        print(f"Error computing average_min_edit_distance: {e}")

def count_broken_feature_mentions(captions: List[str], feature: str) -> float:
    """
    Calculate percentage of captions mentioning a broken feature
    
    Args:
        captions: List of caption strings
        feature: Feature to check ("pipe" or "cannon")
    
    Returns:
        Percentage of captions mentioning broken feature
    """
    # Clean captions by removing "no broken" phrases
    cleaned_captions = [
        caption.replace(f"no broken {feature}s", "").replace(f"no broken {feature}", "")
        for caption in captions
    ]
    
    # Count mentions of broken feature
    broken_count = sum(
        f"broken {feature}" in caption.lower()
        for caption in cleaned_captions
    )
    
    return (broken_count / len(captions)) * 100 if captions else 0.0

def analyze_broken_features_from_data(data: List[Dict], feature: str) -> float:
    """
    Analyze broken features from list of scene/caption dictionaries
    
    Args:
        data: List of dictionaries containing 'caption' keys
        feature: Feature to check ("pipe" or "cannon")
    
    Returns:
        Percentage of scenes with broken feature
    """
    captions = [entry['caption'] for entry in data if 'caption' in entry]
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
    captions = [
        assign_caption(
            scene,
            id_to_char,
            title_descriptors,
            describe_locations=False,
            describe_absence=False
        ) 
        for scene in scenes
    ]
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
    
def analyze_scene_captions_from_json(json_path: str, feature: str) -> float:
    """
    Analyze broken features in scenes from a JSON file containing scene/caption pairs
    
    Args:
        json_path: Path to JSON file containing list of scene/caption dictionaries
        feature: Feature to check ("pipe" or "cannon")
    
    Returns:
        Percentage of scenes with broken feature
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        KeyError: If JSON entries don't contain 'caption' field
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of scene/caption dictionaries")
            
        # Extract captions, skipping entries without captions
        captions = [entry['caption'] for entry in data if 'caption' in entry]
        
        if not captions:
            print(f"Warning: No captions found in {json_path}")
            return 0.0
            
        return count_broken_feature_mentions(captions, feature)
        
    except FileNotFoundError:
        print(f"Error: File {json_path} not found")
        raise
    except json.JSONDecodeError:
        print(f"Error: File {json_path} is not valid JSON")
        raise
    except KeyError as e:
        print(f"Error: Invalid data format - missing caption field")
        raise
    
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

# TODO: implement strict and non-strict phrase targeting metrics
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

# TODO: GitHub Issue #56 - A* Solvability
def astar_metrics():
    """
     Takes a list of levels in the list of strings format used by MarioGPT. 
     For each of the levels, run my astar code on it (repeat some number of times specified by a parameter). 
     For each run, parse the returned string to extract information about performance. 
     Average the performance of he A* agent across multiple runs.
      Return a list of organized results indicating how A* performed on each level (a list of dictionaries).
     """
    pass  # Placeholder for future A* metrics implementation

if __name__ == "__main__":
    # Base directory for datasets
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#     # List of all datasets to analyze
#     datasets = [
#         ("SMB1", "SMB1_LevelsAndCaptions-regular.json"),
#         ("SMB2", "SMB2_LevelsAndCaptions-regular.json"),
#         ("SML", "SML_LevelsAndCaptions-regular.json"),
#         ("SMB1AND2", "SMB1AND2_LevelsAndCaptions-regular.json"),
#         ("Mario-All", "Mario_LevelsAndCaptions-regular.json")
#     ]
#     #TESTING FOR EDIT DISTANCE #61
#     print("Analyzing Edit Distances Across Datasets")
#     print("=" * 50)
    
#     """
#     Expected Results (based on previous runs):
    
#     Super Mario Bros 1:
#     - Average Edit Distance: ~10.1
    
#     Super Mario Bros 2:
#     - Average Edit Distance: ~11.3
    
#     Super Mario Land:
#     - Average Edit Distance: ~14.6

#     Combined SMB1+2:
#     - Average Edit Distance: ~10.6
    
#     All Mario Games:
#     - Average Edit Distance: ~11.6
#     """
    
#     for game_name, dataset_file in datasets:
#         dataset_path = os.path.join(base_dir, dataset_file)
#         try:
#             print(f"\nAnalyzing {game_name} levels:")
#             print("-" * 30)

#             # Load dataset
#             with open(dataset_path, 'r') as f:
#                 data = json.load(f)
#                 levels = [entry['scene'] for entry in data if 'scene' in entry]

#             print(f"Loaded {len(levels)} levels")

#             # Continue with existing analysis
#             # dist = edit_distance(levels[0], levels[1])
#             # print(f"Edit distance (Level 0 to 1): {dist}")

#             # min_dist = min_edit_distance(levels[0], levels[1:])
#             # print(f"Min edit distance (Level 0): {min_dist}")

#             avg_dist = average_min_edit_distance(levels)
#             print(f"Average minimum edit distance: {avg_dist:.1f}")

#         except FileNotFoundError:
#             print(f"Dataset file not found: {dataset_file}")
#         except Exception as e:
#             print(f"Error processing {game_name}: {str(e)}\n")
#             print("Stack trace:")
#             import traceback
#             traceback.print_exc()
            
            
    # Test phrase targeting analysis
    test_phrases = [
        "full floor",
        "several enemies",
        "one cannon",
        "one pipe",
        "coin",
        "tower"
    ]
    
    try:
        # Load dataset
        dataset_path = os.path.join(base_dir, "datasets\\SMB1_LevelsAndCaptions-regular.json")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            # Create pairs from data
            pairs = [(entry['caption'], entry['caption']) 
                    for entry in data 
                    if 'caption' in entry]
        
        print(f"\nAnalyzing {len(pairs)} prompt-caption pairs")
        
        for phrase in test_phrases:
            print(f"\n{'-' * 30}")
            print(f"Analyzing phrase: '{phrase}' (Strict Mode)")
            
            # Strict mode
            metrics_strict = calculate_phrase_metrics(pairs, phrase, strict=True)
            print(f"Strict Mode Results:")
            print(f"Total samples analyzed: {metrics_strict['total']}")
            print(f"True Positives:  {metrics_strict['true_positives']} ({metrics_strict['true_positives']/metrics_strict['total']*100:.1f}%)")
            print(f"False Positives: {metrics_strict['false_positives']} ({metrics_strict['false_positives']/metrics_strict['total']*100:.1f}%)")
            print(f"True Negatives:  {metrics_strict['true_negatives']} ({metrics_strict['true_negatives']/metrics_strict['total']*100:.1f}%)")
            print(f"False Negatives: {metrics_strict['false_negatives']} ({metrics_strict['false_negatives']/metrics_strict['total']*100:.1f}%)")
            print(f"Precision: {metrics_strict['precision']:.3f}")
            print(f"Recall:    {metrics_strict['recall']:.3f}")
            print(f"F1 Score:  {metrics_strict['f1_score']:.3f}")
            
            print(f"\nAnalyzing phrase: '{phrase}' (Non-Strict Mode)")
            
            # Non-strict mode
            metrics_non_strict = calculate_phrase_metrics(pairs, phrase, strict=False)
            print(f"Non-Strict Mode Results:")
            print(f"Total samples analyzed: {metrics_non_strict['total']}")
            print(f"True Positives:  {metrics_non_strict['true_positives']} ({metrics_non_strict['true_positives']/metrics_non_strict['total']*100:.1f}%)")
            print(f"False Positives: {metrics_non_strict['false_positives']} ({metrics_non_strict['false_positives']/metrics_non_strict['total']*100:.1f}%)")
            print(f"True Negatives:  {metrics_non_strict['true_negatives']} ({metrics_non_strict['true_negatives']/metrics_non_strict['total']*100:.1f}%)")
            print(f"False Negatives: {metrics_non_strict['false_negatives']} ({metrics_non_strict['false_negatives']/metrics_non_strict['total']*100:.1f}%)")
            print(f"Precision: {metrics_non_strict['precision']:.3f}")
            print(f"Recall:    {metrics_non_strict['recall']:.3f}")
            print(f"F1 Score:  {metrics_non_strict['f1_score']:.3f}")
            
    except Exception as e:
        print(f"Error in phrase targeting analysis: {str(e)}")
        traceback.print_exc()