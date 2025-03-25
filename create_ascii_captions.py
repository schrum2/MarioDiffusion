import json
import sys
import os

def get_tile_descriptors(tileset):
    """Creates a mapping from tile character to its list of descriptors."""
    return {char: set(attrs) for char, attrs in tileset["tiles"].items()}

def analyze_floor(scene, id_to_char, tile_descriptors):
    """Analyzes the last row of the 16x16 scene and generates a floor description."""
    last_row = scene[-1]  # The bottom row of the scene
    solid_count = sum(1 for tile in last_row if "solid" in tile_descriptors.get(id_to_char[tile], []))
    passable_count = sum(1 for tile in last_row if "passable" in tile_descriptors.get(id_to_char[tile], []))

    if solid_count == 16:
        return "full floor"
    elif passable_count == 16:
        return "no floor"
    elif solid_count > passable_count:
        # Count contiguous groups of passable tiles
        gaps = 0
        in_gap = False
        for tile in last_row:
            if "passable" in tile_descriptors.get(id_to_char[tile], []):
                if not in_gap:
                    gaps += 1
                    in_gap = True
            elif "solid" in tile_descriptors.get(id_to_char[tile], []):
                in_gap = False
            else:
                print("error")
                print(tile)
                print(tile_descriptors)
                print(tile_descriptors.get(tile, []))
                raise ValueError("Every tile should be either passable or solid")
        return f"floor with {gaps} gap" + ("s" if gaps > 1 else "")
    else:
        # Count contiguous groups of solid tiles
        chunks = 0
        in_chunk = False
        for tile in last_row:
            if "solid" in tile_descriptors.get(id_to_char[tile], []):
                if not in_chunk:
                    chunks += 1
                    in_chunk = True
            elif "passable" in tile_descriptors.get(id_to_char[tile], []):
                in_chunk = False
            else:
                print("error")
                print(tile)
                print(tile_descriptors)
                print(tile_descriptors.get(tile, []))
                raise ValueError("Every tile should be either passable or solid")
        return f"giant gap with {chunks} chunk"+("s" if chunks > 1 else "")+" of floor"

def count_in_scene(scene, tiles):
    """ counts standalone tiles """
    count = 0
    for row in scene:
        for t in row: 
            if t in tiles:
                count += 1

    return count

def count_caption_phrase(scene, tiles, name, names, offset = 0):
    """ offset modifies count used in caption """
    count = offset + count_in_scene(scene, tiles)
    if count > 0: 
        return " " + str(count) + " " + (names if count > 1 else name) + "."
    else:
        return ""

def in_column(scene, x, tile):
    for row in scene:
        if row[x] == tile:
            return True

    return False

def analyze_ceiling(scene, id_to_char, tile_descriptors):
    """
    Analyzes row 4 (0-based index) to detect a ceiling.
    Returns a caption phrase or an empty string if no ceiling is detected.
    """
    ceiling_row = 4
    if ceiling_row >= len(scene):
        return ""  # Scene too short to have a ceiling

    row = scene[ceiling_row]
    solid_count = sum(1 for tile in row if "solid" in tile_descriptors.get(id_to_char[tile], []))
    passable_count = sum(1 for tile in row if "passable" in tile_descriptors.get(id_to_char[tile], []))

    if solid_count == 16:
        return " full ceiling."
    elif solid_count > 8:
        # Count contiguous gaps of passable tiles
        gaps = 0
        in_gap = False
        for tile in row:
            if "passable" in tile_descriptors.get(id_to_char[tile], []):
                if not in_gap:
                    gaps += 1
                    in_gap = True
            else:
                in_gap = False
        return f" ceiling with {gaps} gap" + ("s" if gaps > 1 else "") + "."
    else:
        return ""  # Not enough solid tiles for a ceiling

def find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor, min_run_length=2, require_above_below_not_solid=False, exclude_rows = []):
    """
    Finds horizontal lines (runs) of tiles with the target descriptor.
    - Skips the bottom row
    - Skips tiles marked as 'pipe'
    - Can require non-solid space above and below (for platforms)
    Returns a list of (y, start_x, end_x) tuples
    """
    lines = []
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0

    for y in range(height - 1):  # Skip bottom row
        if y in exclude_rows:
            continue # Could skip ceiling

        x = 0
        while x < width:
            tile_char = id_to_char[scene[y][x]]
            descriptors = tile_descriptors.get(tile_char, [])

            if (target_descriptor not in descriptors) or ("pipe" in descriptors):
                x += 1
                continue

            # If required, check for passable tiles above and below
            if require_above_below_not_solid:
                # Above
                if y > 0:
                    above_char = id_to_char[scene[y - 1][x]]
                    if "solid" in tile_descriptors.get(above_char, []):
                        x += 1
                        continue
                else:
                    x += 1
                    continue
                # Below
                if y + 1 < height:
                    below_char = id_to_char[scene[y + 1][x]]
                    if "solid" in tile_descriptors.get(below_char, []):
                        x += 1
                        continue
                else:
                    x += 1
                    continue

            # Start of valid run
            run_start = x
            while x < width:
                tile_char = id_to_char[scene[y][x]]
                descriptors = tile_descriptors.get(tile_char, [])

                if (target_descriptor in descriptors and "pipe" not in descriptors):
                    if require_above_below_not_solid:
                        if y > 0 and "solid" in tile_descriptors.get(id_to_char[scene[y - 1][x]], []):
                            break
                        if y + 1 < height and "solid" in tile_descriptors.get(id_to_char[scene[y + 1][x]], []):
                            break
                    x += 1
                else:
                    break
            run_length = x - run_start
            if run_length >= min_run_length:
                lines.append((y, run_start, x - 1))
    return lines

def describe_horizontal_lines(lines, label):
    """
    Outputs phrases like:
      " 3 platforms at rows 4,7,9."
    """
    if not lines:
        return ""
    # Extract all unique y positions of valid runs
    rows = [y for y, _, _ in lines]
    rows_text = ", ".join(str(y) for y in sorted(rows))
    count = len(rows)
    plural = label + "s" if count > 1 else label
    return f" {count} {plural} at rows {rows_text}."

def analyze_staircases(scene, id_to_char, tile_descriptors):
    """
    Detects staircases in the scene.
    A staircase is a sequence of at least 3 columns where solid tiles form steps increasing by 1 row each.
    Above each solid block must be passable.
    Returns a caption phrase or empty string.
    """
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0
    staircases = 0
    col = 0

    while col <= width - 3:
        # Try to find the start of a staircase
        step_cols = []
        for start_row in range(height - 3):
            if is_staircase_from(scene, id_to_char, tile_descriptors, col, start_row):
                # Now count how many columns this staircase extends
                length = 3
                while col + length < width and is_staircase_from(scene, id_to_char, tile_descriptors, col + length - 2, start_row):
                    length += 1
                staircases += 1
                col += length  # Skip past this staircase
                break  # Restart staircase search from new col
        else:
            col += 1  # No staircase starting here, move right

    if staircases > 0:
        return f" {staircases} staircase{'s' if staircases > 1 else ''}."
    else:
        return ""

def is_staircase_from(scene, id_to_char, tile_descriptors, start_col, start_row):
    """
    Checks if there's a valid 3-step staircase starting at (start_col, start_row).
    """
    try:
        for step in range(3):
            row = start_row + step
            col = start_col + step
            tile = scene[row][col]
            if "solid" not in tile_descriptors.get(id_to_char[tile], []):
                return False
            # Check above this block is passable
            if row > 0:
                tile_above = scene[row - 1][col]
                if "solid" in tile_descriptors.get(id_to_char[tile_above], []):
                    return False
        return True
    except IndexError:
        return False  # Out of bounds means no staircase

def generate_captions(dataset_path, tileset_path, output_path):
    """Processes the dataset and generates captions for each level scene."""
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Load tileset
    with open(tileset_path, "r") as f:
        tileset = json.load(f)
        #print(f"tileset: {tileset}")
        tile_chars = sorted(tileset['tiles'].keys())
        id_to_char = {idx: char for idx, char in enumerate(tile_chars)}
        char_to_id = {char: idx for idx, char in enumerate(tile_chars)}
        tile_descriptors = get_tile_descriptors(tileset)
        #print(f"tile_descriptors: {tile_descriptors}")

    # Generate captions
    captioned_dataset = []
    for scene in dataset:
        caption = analyze_floor(scene, id_to_char, tile_descriptors) + "."
        ceiling = analyze_ceiling(scene, id_to_char, tile_descriptors)
        caption += ceiling

        caption += count_caption_phrase(scene, [char_to_id['E']], "enemy", "enemies")
        caption += count_caption_phrase(scene, [char_to_id['Q'],char_to_id['?']], "question block", "question blocks")
  
        pipe_at_edge = 1 if in_column(scene, 0, char_to_id['>']) else 0
        caption += count_caption_phrase(scene, [char_to_id['<']], "pipe", "pipes", pipe_at_edge)

        caption += count_caption_phrase(scene, [char_to_id['o']], "coin", "coins")
        # Coin lines - no passable/solid requirements
        coin_lines = find_horizontal_lines(
            scene, id_to_char, tile_descriptors, 
            target_descriptor="coin",
            min_run_length=2
        )
        caption += describe_horizontal_lines(coin_lines, "coin line")

        # Platforms - solid tiles with passable above and below, no pipes
        platform_lines = find_horizontal_lines(
            scene, id_to_char, tile_descriptors, 
            target_descriptor="solid",
            min_run_length=2,
            require_above_below_not_solid=True,
            exclude_rows = [] if ceiling == "" else [4] # ceiling is not a platform
        )
        caption += describe_horizontal_lines(platform_lines, "platform")

        caption += analyze_staircases(scene, id_to_char, tile_descriptors)

        captioned_dataset.append({
            "scene": scene,
            "caption": caption
        })

    # Save new dataset with captions
    with open(output_path, "w") as f:
        json.dump(captioned_dataset, f, indent=4)

    print(f"Captioned dataset saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate captions for Mario screenshots")
    parser.add_argument("--dataset", required=True, help="json with level scenes")
    parser.add_argument("--tileset", default='..\TheVGLC\Super Mario Bros\smb.json', help="Descriptions of individual tile types")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")

    args = parser.parse_args()

    dataset_file = args.dataset
    tileset_file = args.tileset
    output_file = args.output

    if not os.path.isfile(dataset_file) or not os.path.isfile(tileset_file):
        print("Error: One or more input files do not exist.")
        sys.exit(1)

    generate_captions(dataset_file, tileset_file, output_file)
