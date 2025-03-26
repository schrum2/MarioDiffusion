import json
import sys
import os

WIDTH = 16
HEIGHT = 16

BOTTOM = 16
CEILING = 4

def get_tile_descriptors(tileset):
    """Creates a mapping from tile character to its list of descriptors."""
    return {char: set(attrs) for char, attrs in tileset["tiles"].items()}

def analyze_floor(scene, id_to_char, tile_descriptors):
    """Analyzes the last row of the 16x16 scene and generates a floor description."""
    last_row = scene[-1]  # The bottom row of the scene
    solid_count = sum(1 for tile in last_row if "solid" in tile_descriptors.get(id_to_char[tile], []))
    passable_count = sum(1 for tile in last_row if "passable" in tile_descriptors.get(id_to_char[tile], []))

    if solid_count == WIDTH:
        return "full floor"
    elif passable_count == WIDTH:
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
    ceiling_row = CEILING 
    if ceiling_row >= len(scene):
        return ""  # Scene too short to have a ceiling

    row = scene[ceiling_row]
    solid_count = sum(1 for tile in row if "solid" in tile_descriptors.get(id_to_char[tile], []))
    passable_count = sum(1 for tile in row if "passable" in tile_descriptors.get(id_to_char[tile], []))

    if solid_count == WIDTH:
        return " full ceiling."
    elif solid_count > WIDTH//2:
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

def find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor, min_run_length=2, require_above_below_not_solid=False, exclude_rows = [], already_accounted = set()):
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
        possible_locations = set()
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

                    possible_locations.add( (y,x) )
                    x += 1
                else:
                    break
            run_length = x - run_start
            if run_length >= min_run_length:
                already_accounted.update(possible_locations) # Blocks of the line are now accounted for
                lines.append((y, run_start, x - 1))

    return lines

def describe_horizontal_lines(lines, label):
    """
    Outputs phrases like:
      "3 platforms at rows 4 (cols 5-10), 7 (cols 3-8), 9 (cols 12-15)."
    """
    if not lines:
        return ""
    
    parts = []
    for y, start_x, end_x in sorted(lines):
        parts.append(f"{y} (cols {start_x}-{end_x})")
    
    count = len(lines)
    plural = label + "s" if count > 1 else label
    return f" {count} {plural} at row{'s' if count > 1 else ''} " + ", ".join(parts) + "."

def analyze_staircases(scene, id_to_char, tile_descriptors, verticality, already_accounted):
    """
    Detects staircases in the scene.
    verticality = 1 for descending, verticality = -1 for ascending
    A staircase is a sequence of at least 3 columns where solid tiles form steps increasing by 1 row each.
    Above each solid block must be passable.
    Returns a caption phrase or empty string.
    """
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0
    staircases = 0
    col = 0
    staircase_lengths = []

    while col <= width - 3:
        # Try to find the start of a staircase
        step_cols = []
        for start_row in range(0 if verticality == 1 else 3, height - 3 if verticality == 1 else height - 1):
            if is_staircase_from(scene, id_to_char, tile_descriptors, col, start_row, verticality, already_accounted):
                # Now count how many columns this staircase extends
                length = 3
                while col + length < width and is_staircase_from(scene, id_to_char, tile_descriptors, col + length - 2, start_row + verticality*(length - 2), verticality, already_accounted):
                    length += 1
                staircases += 1
                col += length  # Skip past this staircase
                staircase_lengths.append(length)
                break  # Restart staircase search from new col
        else:
            col += 1  # No staircase starting here, move right

    type = "descending" if verticality == 1 else "ascending"
    if staircases > 0:
        return f" {staircases} {type} staircase{'s' if staircases > 1 else ''} with length{'s' if staircases > 1 else ''} {', '.join(map(str, staircase_lengths))}."
    else:
        return ""

def is_staircase_from(scene, id_to_char, tile_descriptors, start_col, start_row, verticality, already_accounted):
    """
    Checks if there's a valid 3-step staircase starting at (start_col, start_row).
    verticality = 1 for descending staircase, verticality = -1 for ascending
    """
    try:
        blocks_in_stairs = set()
        for step in range(3):
            row = start_row + verticality*step
            if row == len(scene) - 1: 
                return False # Do not count floor in staircases
            col = start_col + step
            tile = scene[row][col]
            if "solid" not in tile_descriptors.get(id_to_char[tile], []):
                return False
            # Check above this block is passable
            if row > 0:
                tile_above = scene[row - 1][col]
                if "solid" in tile_descriptors.get(id_to_char[tile_above], []):
                    return False
                else:
                    # Blocks beneath the stairs are also part of stairs
                    row2 = row
                    while row2 < len(scene) and "solid" in tile_descriptors.get(id_to_char[scene[row2][col]], []): 
                        blocks_in_stairs.add( (row2,col) )
                        row2 += 1                    

        # Only add all of the blocks once it is confirmed to be a staircase
        already_accounted.update(blocks_in_stairs)
        return True
    except IndexError:
        return False  # Out of bounds means no staircase

def flood_fill(scene, visited, start_row, start_col, id_to_char, tile_descriptors, excluded, pipes=False):
    stack = [(start_row, start_col)]
    structure = []

    while stack:
        row, col = stack.pop()
        if (row, col) in visited or (row, col) in excluded:
            continue
        tile = scene[row][col]
        descriptors = tile_descriptors.get(id_to_char[tile], [])
        if "solid" not in descriptors or (not pipes and "pipe" in descriptors):
            continue

        visited.add((row, col))
        structure.append((row, col))

        # Check neighbors
        for d_row, d_col in [(-1,0), (1,0), (0,-1), (0,1)]:
            n_row, n_col = row + d_row, col + d_col
            if 0 <= n_row < len(scene) and 0 <= n_col < len(scene[0]):
                stack.append((n_row, n_col))

    return structure

def find_solid_structures(scene, id_to_char, tile_descriptors, already_accounted, pipes = False):
    """Find unaccounted solid block structures"""
    visited = set()
    structures = []

    for row in range(len(scene)):
        for col in range(len(scene[0])):
            if (row, col) in visited or (row, col) in already_accounted:
                continue
            tile = scene[row][col]
            descriptors = tile_descriptors.get(id_to_char[tile], [])
            if (not pipes and "solid" in descriptors and "pipe" not in descriptors) or (pipes and "pipe" in descriptors):
                structure = flood_fill(scene, visited, row, col, id_to_char, tile_descriptors, already_accounted, pipes)
                if pipes or len(structure) >= 4:  # Ignore tiny groups of blocks, but keep all pipes
                    structures.append(structure)

    return structures

def describe_structures(structures, ceiling_row=CEILING, pipes=False):
    descriptions = []
    for struct in structures:
        min_row = min(pos[0] for pos in struct)
        max_row = max(pos[0] for pos in struct)
        min_col = min(pos[1] for pos in struct)
        max_col = max(pos[1] for pos in struct)

        width = max_col - min_col + 1
        height = max_row - min_row + 1

        if pipes:
            desc = "pipe"
        else:
            attached_to_ceiling = any(r == ceiling_row for r, c in struct)

            if width <= 2 and height >= 4:
                desc = "tall tower"
            elif width >= 4 and height <= 2:
                desc = "wide wall"
            else:
                desc = "irregular block cluster"

            if attached_to_ceiling:
                desc += " attached to the ceiling"

        desc += f" from row {min_row} to {max_row}, columns {min_col} to {max_col}"
        descriptions.append(desc)
    
    if descriptions:
        return " " + "; ".join(descriptions) + "."
    return ""

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
        already_accounted = set()
        # Include all of floor, even empty tiles
        for x in range(WIDTH):
            already_accounted.add( (BOTTOM - 1,x) )

        caption = analyze_floor(scene, id_to_char, tile_descriptors) + "."
        ceiling = analyze_ceiling(scene, id_to_char, tile_descriptors)
        caption += ceiling

        if ceiling != "":
            # Include all of ceiling, even empty tiles
            for x in range(WIDTH):
                already_accounted.add( (CEILING,x) )
        
        caption += count_caption_phrase(scene, [char_to_id['E']], "enemy", "enemies")
        caption += count_caption_phrase(scene, [char_to_id['Q'],char_to_id['?']], "question block", "question blocks")
  
        # Did not give pipe position information
        #pipe_at_edge = 1 if in_column(scene, 0, char_to_id['>']) else 0
        #caption += count_caption_phrase(scene, [char_to_id['<']], "pipe", "pipes", pipe_at_edge)

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
            exclude_rows = [] if ceiling == "" else [4], # ceiling is not a platform
            already_accounted=already_accounted
        )
        caption += describe_horizontal_lines(platform_lines, "platform")

        caption += analyze_staircases(scene, id_to_char, tile_descriptors, -1, already_accounted=already_accounted)
        caption += analyze_staircases(scene, id_to_char, tile_descriptors, 1, already_accounted=already_accounted)

        structures = find_solid_structures(scene, id_to_char, tile_descriptors, already_accounted, pipes=True)
        caption += describe_structures(structures, pipes=True)

        structures = find_solid_structures(scene, id_to_char, tile_descriptors, already_accounted)
        caption += describe_structures(structures)

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
