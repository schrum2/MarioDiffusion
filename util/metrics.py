"""
This module provides utility functions for comparing level layouts through various metrics.

The functions in this module operate on level layouts represented as 2D lists/arrays
where each element represents a tile. The specific tile representation can be arbitrary
(characters, integers, etc.) as long as equality comparison is supported between tiles.

Results from running on different datasets:
SMB1_Levels.json - Original Super Mario Bros levels
SMB2_Levels.json - Super Mario Bros 2 (Japan) levels
SML_Levels.json - Super Mario Land levels
Mario_Levels.json - Combined Mario levels from all games
SMB1AND2_Levels.json - Combined levels from SMB1 and SMB2
"""

from typing import List, Sequence, TypeVar, Union
import numpy as np
import json
from pathlib import Path

# Type variable for the tile type
T = TypeVar('T')

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

def average_min_edit_distance(levels: Sequence[Sequence[Sequence[T]]]) -> float:
    """
    Calculate the average minimum edit distance across all levels in a dataset.
    
    For each level, finds its minimum edit distance to any other level in the dataset,
    then averages these minimum distances.
    
    Args:
        levels: A sequence of level layouts
        
    Returns:
        The average of the minimum edit distances
        
    Raises:
        ValueError: If fewer than 2 levels are provided or if levels have different dimensions
    """
    if len(levels) < 2:
        raise ValueError("Need at least 2 levels to compute average minimum edit distance")
    
    # For each level, compute its min edit distance to all other levels
    min_distances = []
    for i, level in enumerate(levels):
        # Create list of all levels except the current one
        other_levels = levels[:i] + levels[i+1:]
        min_dist = min_edit_distance(level, other_levels)
        min_distances.append(min_dist)
    
    return sum(min_distances) / len(min_distances)

def process_dataset(dataset_path: str) -> None:
    """Process a dataset and print its metrics."""
    try:
        with open(dataset_path, "r") as f:
            levels = json.load(f)
        
        if not levels:
            print(f"{dataset_path}: Empty dataset")
            return
            
        # Clean up any empty lists or malformed data
        levels = [level for level in levels if level and all(row for row in level)]
        
        if len(levels) < 2:
            print(f"{dataset_path}: Not enough valid levels (need at least 2)")
            return
            
        avg_dist = average_min_edit_distance(levels)
        print(f"\nResults for {Path(dataset_path).name}:")
        print(f"Number of levels: {len(levels)}")
        print(f"Average minimum edit distance: {avg_dist:.2f}")
        if len(levels) >= 2:
            print(f"Example edit distance between first two levels: {edit_distance(levels[0], levels[1])}")
            
    except Exception as e:
        print(f"Error processing {dataset_path}: {str(e)}")

if __name__ == "__main__":
    # Process all datasets
    datasets = [
        "SMB1_Levels.json",
        "SMB2_Levels.json", 
        "SML_Levels.json",
        "Mario_Levels.json",
        "SMB1AND2_Levels.json"
    ]
    
    current_dir = Path(__file__).parent.parent
    for dataset in datasets:
        dataset_path = current_dir / dataset
        if dataset_path.exists():
            process_dataset(str(dataset_path))
        else:
            print(f"\nWarning: {dataset} not found")
