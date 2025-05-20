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
    extra_tile = '!'
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

def level_to_id_grid(level, tile_to_id):
    width = max(len(row) for row in level)
    height = len(level)
    target_size = max(width, height)
    # Pad each row to the target size
    padded_level = [row.ljust(target_size, extra_tile) for row in level]
    # Pad rows to the target size
    while len(padded_level) < target_size:
        padded_level.append(extra_tile * target_size)
    # Convert to grid of tile IDs
    return [[tile_to_id.get(c, tile_to_id[extra_tile]) for c in row] for row in padded_level]

def main(tileset_path, levels_dir, output_path):
    tile_to_id = load_tileset(tileset_path)
    levels = load_levels(levels_dir)
    
    dataset = []
    for level in levels:
        #samples = pad_and_sample(level, tile_to_id)
        #dataset.extend(samples)
        grid = level_to_id_grid(level, tile_to_id)
        dataset.append(grid)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tileset', default='..\TheVGLC\Super Mario Bros\smb.json', help='Path to the tile set JSON')
    parser.add_argument('--levels', default='..\TheVGLC\Super Mario Bros\Processed', help='Directory containing level text files')
    parser.add_argument('--output', required=True, help='Path to the output JSON file')
    args = parser.parse_args()
    main(args.tileset, args.levels, args.output)
