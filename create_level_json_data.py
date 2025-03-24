import os
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
    levels = []
    for file in sorted(Path(levels_dir).glob("*.txt")):
        with open(file, 'r') as f:
            level = [line.strip() for line in f if line.strip()]
            levels.append(level)
    return levels

def pad_and_sample(level, tile_to_id):
    height = len(level)
    width = max(len(row) for row in level)
    pad_rows = 16 - height
    padded_level = ["-" * width] * pad_rows + level

    # Ensure each row is the same width
    padded_level = [row.ljust(width, '-') for row in padded_level]
    
    samples = []
    for x in range(width - 16 + 1):
        sample = []
        for y in range(16):
            window_row = padded_level[y][x:x+16]
            sample.append([tile_to_id.get(c, tile_to_id['-']) for c in window_row])
        samples.append(sample)
    return samples

def main(tileset_path, levels_dir, output_path):
    tile_to_id = load_tileset(tileset_path)
    levels = load_levels(levels_dir)
    
    dataset = []
    for level in levels:
        samples = pad_and_sample(level, tile_to_id)
        dataset.extend(samples)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tileset', default='..\TheVGLC\Super Mario Bros\smb.json', help='Path to the tile set JSON')
    parser.add_argument('--levels', default='..\TheVGLC\Super Mario Bros\Processed', help='Directory containing level text files')
    parser.add_argument('--output', required=True, help='Path to the output JSON file')
    args = parser.parse_args()
    main(args.tileset, args.levels, args.output)
