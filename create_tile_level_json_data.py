import json
import argparse
from pathlib import Path

def load_tileset(tileset_path):
    """
    Loads a tileset from a JSON file and maps tile characters to unique IDs.

    Args:
        tileset_path (str): Path to the JSON file containing the tileset data.

    Returns:
        dict: A dictionary mapping tile characters to unique integer IDs.
    """
    with open(tileset_path, 'r') as f:
        tileset_data = json.load(f)
    tile_chars = sorted(tileset_data['tiles'].keys())
    tile_to_id = {char: idx for idx, char in enumerate(tile_chars)}
    return tile_to_id

def load_levels(levels_dir):
    """
    Loads levels from text files in a specified directory.

    Args:
        levels_dir (str): Path to the directory containing level text files.

    Returns:
        list: A list of levels, where each level is represented as a list of strings.
    """
    levels = set()
    for file in sorted(Path(levels_dir).glob("*.txt")):
        with open(file, 'r') as f:
            level = tuple(line.strip() for line in f if line.strip())
            # if level:  # Only add non-empty levels
            #     print(f"Loaded level from {file.name} with {len(level)} rows")
            #     for row in level:
            #         print(f"  {row}")
            levels.add(level)
    return [list(level) for level in levels]  # Convert back to list-of-lists for compatibility


def pad_and_sample(level, tile_to_id, window_size):
    """
    Extracts tile samples of a specified window size from a level.

    Args:
        level (list): A 2D list representing the level layout.
        tile_to_id (dict): A dictionary mapping tile characters to unique IDs.
        window_size (int): The size of the square window to extract.

    Returns:
        list: A list of 2D lists, each representing a sampled window of tiles.
    """
    height = len(level)
    width = len(level[0])
    samples = set()  # Use a set to avoid duplicates

    # Iterate through the level, extracting tiles that fit entirely within bounds
    for y in range(0, height - window_size + 1):
        for x in range(0, width - window_size + 1):
            sample = []
            for row_idx in range(y, y + window_size):
                window_row = []
                for col_idx in range(x, x + window_size):
                    char = level[row_idx][col_idx]
                    tile_id = tile_to_id.get(char, -1)
                    window_row.append(tile_id)
                sample.append(tuple(window_row))  # Convert row to tuple
            samples.add(tuple(sample))  # Convert sample to tuple of tuples before adding

    print(f"Extracted {len(samples)} unique samples.")
    return samples

def main(tileset_path, levels_dir, output_path, window_size):
    """
    Orchestrates the process of loading tilesets and levels, generating samples, and saving them to a JSON file.

    Args:
        tileset_path (str): Path to the JSON file containing the tileset data.
        levels_dir (str): Path to the directory containing level text files.
        output_path (str): Path to save the output JSON file.
        window_size (int): The size of the square window to extract.

    Returns:
        None
    """
    tile_to_id = load_tileset(tileset_path)
    levels = load_levels(levels_dir)
    
    dataset = []
    unique_set = set()
    for level in levels:
        samples = pad_and_sample(level, tile_to_id, window_size)
        for sample in samples:
            dataset.append([list(row) for row in sample])
            unique_set.add(sample)

    print(f"Total samples: {len(dataset)}")
    print(f"Unique samples: {len(unique_set)}")

    
    # Convert back to lists for JSON serialization
    dataset = [ [list(row) for row in sample] for sample in unique_set ]

    print(f"Dataset elements: {len(dataset)}")

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Reload the saved JSON file and print the length of the loaded list
    with open(output_path, 'r') as f:
        loaded_dataset = json.load(f)
    print(f"Length of loaded dataset: {len(loaded_dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tileset', default='..\TheVGLC\Super Mario Bros\smb.json', help='Path to the tile set JSON')
    parser.add_argument('--levels', default='..\TheVGLC\Super Mario Bros\Processed', help='Directory containing level text files')
    parser.add_argument('--output', required=True, help='Path to the output JSON file')
    parser.add_argument('--tile_size', type=int, required=False, help='Size of the tile (window) to extract')
    args = parser.parse_args()

    # Add debug prints
    print(f"Loading tileset from: {args.tileset}")
    print(f"Loading levels from: {args.levels}")
    print(f"Output will be saved to: {args.output}")
    print(f"Using tile size: {args.tile_size}")

    # Call main with parsed arguments
    main(args.tileset, args.levels, args.output, args.tile_size)