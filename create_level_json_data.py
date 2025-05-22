import json
import argparse
from pathlib import Path

def load_tileset(tileset_path):
    with open(tileset_path, 'r') as f:
        tileset_data = json.load(f)
    tile_chars = sorted(tileset_data['tiles'].keys())
    # ! is used to indicate the outside of actual level
    # and is not a tile in the tileset
    global extra_tile
    extra_tile = '%'
    if extra_tile not in tile_chars:
        tile_chars.append(extra_tile)
    tile_to_id = {char: idx for idx, char in enumerate(tile_chars)}
    return tile_to_id

def load_levels(levels_dir):
    levels = []
    for file in sorted(Path(levels_dir).glob("*.txt")):
        with open(file, 'r') as f:
            level = [line.strip() for line in f if line.strip()]
            levels.append(level)
    return levels

def pad_and_sample(level, tile_to_id, target_height, target_width):
    height = len(level)
    width = max(len(row) for row in level)
    pad_rows = target_height - height
    padded_level = [extra_tile * width] * pad_rows + level

    # Ensure each row is the same width
    padded_level = [row.ljust(target_width, extra_tile) for row in padded_level]

    samples = []
    for x in range(width - target_width + 1):
        sample = []
        for y in range(target_height):
            window_row = padded_level[y][x:x+target_width]
            sample.append([tile_to_id.get(c, tile_to_id[extra_tile]) for c in window_row])
        samples.append(sample)
    return samples

def main(tileset_path, levels_dir, output_path, target_height, target_width):
    tile_to_id = load_tileset(tileset_path)
    levels = load_levels(levels_dir)
    
    dataset = []
    for level in levels:
        samples = pad_and_sample(level, tile_to_id, target_height, target_width)
        dataset.extend(samples)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

#Hi bess
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tileset', default='..\\TheVGLC\\Super Mario Bros\\smb.json', help='Path to the tile set JSON')
    parser.add_argument('--levels', default='..\\TheVGLC\\Super Mario Bros\\Processed', help='Directory containing level text files')
    parser.add_argument('--output', required=True, help='Path to the output JSON file')
    parser.add_argument('--target_height', type=int, default=16, help='Target output height (e.g., 16 or 14)')
    parser.add_argument('--target_width', type=int, default=16, help='Target output width (e.g., 16)')
    parser.add_argument('--extra_tile', default='%', help='Padding tile character (should not be a real tile)')
    args = parser.parse_args()
    global extra_tile
    extra_tile = args.extra_tile
    main(args.tileset, args.levels, args.output, args.target_height, args.target_width)
