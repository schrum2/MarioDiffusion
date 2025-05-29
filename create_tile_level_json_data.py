import json
import argparse
from pathlib import Path

def load_tileset(tileset_path):
    with open(tileset_path, 'r') as f:
        tileset_data = json.load(f)
    tile_chars = sorted(tileset_data['tiles'].keys())
    tile_to_id = {char: idx for idx, char in enumerate(tile_chars)}
    return tile_to_id

def load_levels(levels_dir):
    levels = set()
    for file in sorted(Path(levels_dir).glob("*.txt")):
        with open(file, 'r') as f:
            level = tuple(line.strip() for line in f if line.strip())
            if level:  # Only add non-empty levels
                levels.add(level)
    return [list(level) for level in levels]  # Convert back to list-of-lists for compatibility


def pad_and_sample(level, tile_to_id, window_size):
    height = len(level)
    width = max(len(row) for row in level)
    samples = []
    
    # Iterate through the level, extracting tiles that fit entirely within bounds
    for y in range(0, height - window_size + 1):
        for x in range(0, width - window_size + 1):
            sample = []
            for row_idx in range(y, y + window_size):
                window_row = []
                for col_idx in range(x, x + window_size):
                    window_row.append(tile_to_id.get(level[row_idx][col_idx], -1))
                sample.append(window_row)
            samples.append(sample)
        
    return samples

def main(tileset_path, levels_dir, output_path, window_size):
    tile_to_id = load_tileset(tileset_path)
    levels = load_levels(levels_dir)
    
    sample_set = set()
    for level in levels:
        samples = pad_and_sample(level, tile_to_id, window_size)
        for sample in samples:
            sample_tuple = tuple(tuple(row) for row in sample)  # make hashable
            sample_set.add(sample_tuple)
    
    # Convert back to lists for JSON serialization
    dataset = [ [list(row) for row in sample] for sample in sample_set ]

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)


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