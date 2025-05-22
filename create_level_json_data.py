import json
import argparse
from pathlib import Path

"""
Loads a tileset JSON file (which defines what each tile character means).
Sorts the tile characters and assigns each a unique integer ID.
Adds a special "extra tile" (default %) for padding, if not already present.
Returns a mapping from tile character to integer ID.
"""
def load_tileset(tileset_path):
    with open(tileset_path, 'r') as f:
        tileset_data = json.load(f)
    tile_chars = sorted(tileset_data['tiles'].keys())
    # ! is used to indicate the outside of actual level
    # and is not a tile in the tileset
    global extra_tile
    #extra_tile = '-'
    if extra_tile not in tile_chars:
        tile_chars.append(extra_tile)
    tile_to_id = {char: idx for idx, char in enumerate(tile_chars)}
    return tile_to_id

"""
Reads all .txt files in the given directory.
Each file is a level, each line is a row of tiles.
Strips whitespace and ignores empty lines.
Returns a list of levels, where each level is a list of strings.
"""
def load_levels(levels_dir):
    levels = []
    for file in sorted(Path(levels_dir).glob("*.txt")):
        with open(file, 'r') as f:
            level = [line.strip() for line in f if line.strip()]
            levels.append(level)
    return levels

"""
Extracts room clusters from the level based on the specified room dimensions and window size.
Pads the dungeon grid with void rooms if needed.
Returns a list of room clusters (each is a 3D grid of tile IDs).
"""
def room_cluster_samples(
    level,
    tile_to_id,
    room_width=11,
    room_height=16,
    window_rooms_w=2,
    window_rooms_h=2
):
    # Split the level into a grid of rooms
    dungeon_rows = len(level) // room_height
    dungeon_cols = len(level[0]) // room_width

    # Pad the dungeon grid with void rooms if needed
    padded_rows = dungeon_rows + (window_rooms_h - 1)
    padded_cols = dungeon_cols + (window_rooms_w - 1)
    padded_level = [row.ljust(padded_cols * room_width, extra_tile) for row in level]
    while len(padded_level) < padded_rows * room_height:
        padded_level.append(extra_tile * (padded_cols * room_width))

    print(f"Level size: {len(level)}x{len(level[0])}")
    print(f"Room size: {room_height}x{room_width}")
    print(f"Dungeon grid: {dungeon_rows} rows x {dungeon_cols} cols")
    print(f"Padded grid: {padded_rows} rows x {padded_cols} cols")
    print(f"Padded level height: {len(padded_level)}, width: {len(padded_level[0])}")

    samples = []
    for grid_y in range(padded_rows - window_rooms_h + 1):
        for grid_x in range(padded_cols - window_rooms_w + 1):
            cluster = []
            for wy in range(window_rooms_h):
                for wx in range(window_rooms_w):
                    room_top = (grid_y + wy) * room_height
                    room_left = (grid_x + wx) * room_width
                    room = []
                    for y in range(room_height):
                        row = padded_level[room_top + y][room_left:room_left + room_width]
                        room.append([tile_to_id.get(c, tile_to_id[extra_tile]) for c in row])
                    cluster.append(room)
            # Only keep clusters with at least one non-void room
            if any(any(tile != tile_to_id[extra_tile] for row in room for tile in row) for room in cluster):
                samples.append(cluster)
            else:
                print("Discarded empty cluster")

    print(f"Extracted {len(samples)} clusters")
    if samples:
        print("Sample cluster shape:", len(samples[0]), "rooms,", len(samples[0][0]), "rows per room,", len(samples[0][0][0]), "cols per room")
    return samples

"""
Pads the level with extra rows (using the extra tile) to reach the target height.
Pads each row to the target width.
Slides a window of size target_height x target_width across the level horizontally, 
extracting all possible samples.
Converts each character in the sample window to its tile ID.
Returns a list of samples (each is a 2D grid of tile IDs).
"""
def pad_and_sample(
    level,
    tile_to_id,
    target_height,
    target_width,
    scan_mode="platformer",
    room_width=11,
    room_height=16,
    window_rooms_w=2,
    window_rooms_h=2
):
    if scan_mode == "room_cluster":
        return room_cluster_samples(
            level,
            tile_to_id,
            room_width=room_width,
            room_height=room_height,
            window_rooms_w=window_rooms_w,
            window_rooms_h=window_rooms_h
        )
    else:
        # Platformer: original sliding window
        height = len(level)
        width = max(len(row) for row in level)
        pad_rows = target_height - height
        padded_level = [extra_tile * width] * pad_rows + level
        padded_level = [row.ljust(target_width, extra_tile) for row in padded_level]
        samples = []
        for x in range(width - target_width + 1):
            sample = []
            for y in range(target_height):
                window_row = padded_level[y][x:x+target_width]
                sample.append([tile_to_id.get(c, tile_to_id[extra_tile]) for c in window_row])
            samples.append(sample)
        return samples

"""
Loads the tileset and levels.
For each level, extracts all possible samples.
Collects all samples into a dataset.
Writes the dataset to a JSON file.
"""
def main(
    tileset_path,
    levels_dir,
    output_path,
    target_height,
    target_width,
    scan_mode="platformer",
    room_width=11,
    room_height=16,
    window_rooms_w=2,
    window_rooms_h=2
):
    tile_to_id = load_tileset(tileset_path)
    levels = load_levels(levels_dir)
    dataset = []
    for level in levels:
        samples = pad_and_sample(
            level,
            tile_to_id,
            target_height,
            target_width,
            scan_mode=scan_mode,
            room_width=room_width,
            room_height=room_height,
            window_rooms_w=window_rooms_w,
            window_rooms_h=window_rooms_h
        )
        dataset.extend(samples)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

"""
Allows you to specify the tileset, levels directory, output file, sample size, 
and padding tile from the command line.
Sets the global extra_tile variable based on the argument.
Calls the main function with the provided arguments.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tileset', default='..\\TheVGLC\\Super Mario Bros\\smb.json', help='Path to the tile set JSON')
    parser.add_argument('--levels', default='..\\TheVGLC\\Super Mario Bros\\Processed', help='Directory containing level text files')
    parser.add_argument('--output', required=True, help='Path to the output JSON file')

    parser.add_argument('--target_height', type=int, default=16, help='Target output height (e.g., 16 or 14)')
    parser.add_argument('--target_width', type=int, default=16, help='Target output width (e.g., 16)')
    parser.add_argument('--extra_tile', default='-', help='Padding tile character (should not be a real tile)')
    parser.add_argument('--scan_mode', default='platformer', choices=['platformer', 'room_cluster'], help='Sampling mode')
    parser.add_argument('--room_width', type=int, default=11, help='Room width (room_cluster mode)')
    parser.add_argument('--room_height', type=int, default=16, help='Room height (room_cluster mode)')
    parser.add_argument('--window_rooms_w', type=int, default=2, help='Window width = total number of rooms in window horizantally (room_cluster mode)')
    parser.add_argument('--window_rooms_h', type=int, default=2, help='Window height = total number of vert rooms in window vertically (room_cluster mode)')

    args = parser.parse_args()
    global extra_tile
    extra_tile = args.extra_tile
    main(
        args.tileset,
        args.levels,
        args.output,
        args.target_height,
        args.target_width,
        scan_mode=args.scan_mode,
        room_width=args.room_width,
        room_height=args.room_height,
        window_rooms_w=args.window_rooms_w,
        window_rooms_h=args.window_rooms_h
    )
