"""
This module provides utility functions for comparing level layouts through various metrics.

The functions in this module operate on level layouts represented as 2D lists/arrays
where each element represents a tile. The specific tile representation can be arbitrary
(characters, integers, etc.) as long as equality comparison is supported between tiles.
"""

from typing import List, Dict, Sequence, TypeVar, Union
import sys
import os

# Add the parent directory to the system path to import the extract_tileset function
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        distances = [edit_distance(level, other) for other in level_collection]
        return min(distances)
    except ValueError as e:
        raise ValueError("All levels in collection must have same dimensions as input level") from e

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

if __name__ == "__main__":
    # Test the metrics functions
    import os

    # Use absolute paths for datasets in the MarioDiffusion directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets = [
        os.path.join(base_dir, "broken_pipes.json"),
        os.path.join(base_dir, "SMB2_LevelsAndCaptions-regular.json"),
        os.path.join(base_dir, "SML_LevelsAndCaptions-regular.json"),
    ]

    print("Analyzing broken features:")
    print("-" * 30)
    
    for dataset in datasets:
        try:
            print(f"\nAnalyzing dataset: {dataset}")

            # Load and verify data
            with open(dataset, 'r') as f:
                data = json.load(f)
            print(f"Found {len(data)} entries in dataset")
            
            # Print sample captions for debugging
            print("\nSample captions:")
            for entry in data[:3]:
                if 'caption' in entry:
                    print(f"- {entry['caption']}")

            # Check for broken pipes
            pipe_percentage = analyze_scene_captions_from_json(dataset, "pipe")
            print(f"{dataset} - Broken pipes: {pipe_percentage:.1f}%")
            
            # Check for broken cannons
            cannon_percentage = analyze_scene_captions_from_json(dataset, "cannon")
            print(f"{dataset} - Broken cannons: {cannon_percentage:.1f}%")
            
        except FileNotFoundError:
            print(f"{dataset}: File not found")
        except Exception as e:
            print(f"{dataset}: Error - {str(e)}")
        print()