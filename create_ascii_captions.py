import json
import sys
import os
from collections import Counter
from captions.util import extract_tileset, describe_size, describe_quantity, get_tile_descriptors, analyze_floor, count_in_scene, count_caption_phrase, in_column, analyze_ceiling, flood_fill

import util.common_settings as common_settings

# The floor is the last row of the scene (0-indexed)
FLOOR = common_settings.MARIO_HEIGHT - 1
CEILING = common_settings.MARIO_HEIGHT - 12 #  4

# This is used for describing locations, but it doesn't work well
STANDARD_WIDTH = common_settings.MARIO_WIDTH

LEFT = STANDARD_WIDTH / 3
RIGHT = STANDARD_WIDTH - LEFT

TOP = (FLOOR - CEILING) / 3 + CEILING
BOTTOM = FLOOR - ((FLOOR - CEILING) / 3)

# Could define these via the command line, but for now they are hardcoded
coarse_locations = True
coarse_counts = True
pluralize = True
give_staircase_lengths = False


def describe_location(x, y):
    """
        Describes the location of a point in the scene.
        Returns a string like "left top", "center middle", "right bottom".
        x is the column index, y is the row index.
    """

    if x < LEFT:
        x_desc = "left"
    elif x < RIGHT:
        x_desc = "center"
    else:
        x_desc = "right"

    if y < TOP:
        y_desc = "top"
    elif y < BOTTOM:
        y_desc = "middle"
    else:
        y_desc = "bottom"

    return f"{x_desc} {y_desc}"

def describe_broken_cannons(scene, char_to_id):
    count = 0
    for r in range(len(scene)):
        for c in range(len(scene[r])):
            if scene[r][c] == char_to_id['b']:
                # Although it looks weird, it is ok for either B or b to be above a b tile.
                # The repeated use of b looks weird in visuzalization, but is perfectly valid
                # and looks good in the real game.
                if r == 0 or (scene[r-1][c] != char_to_id['B'] and scene[r-1][c] != char_to_id['b']):
                    count += 1

    if count > 0:
        return f" {describe_quantity(count) if coarse_counts else count} broken " + ("cannons" if pluralize and count > 1 else "cannon") + "."
    else:
        return ""

def find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor, min_run_length=2, require_above_below_not_solid=False, exclude_rows = [], already_accounted = set()):
    """
    Finds horizontal lines (runs) of tiles with the target descriptor.
    - Skips the FLOOR row
    - Skips tiles marked as 'pipe'
    - Can require non-solid space above and below (for platforms)
    - exclude_rows may not be needed because of the alread_accounted set
    Returns a list of (y, start_x, end_x) tuples
    """
    lines = []
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0

    #print((10,0) in already_accounted)

    for y in range(height - 1):  # Skip FLOOR row
        
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
            possible_locations = set()
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

def describe_horizontal_lines(lines, label, describe_locations, describe_absence):
    if not lines:
        if describe_absence:
            return f" no {label}s."
        else:
            return ""
        
    if describe_locations:
        
        if coarse_locations:
            location_counts = {}
            for y, start_x, end_x in sorted(lines):
                location_str = f"{describe_location((end_x + start_x)/2.0, y)}"
                if location_str in location_counts:
                    location_counts[location_str] += 1
                else:
                    location_counts[location_str] = 1

            return " " + ". ".join([f"{describe_quantity(count) if coarse_counts else count} {label}{'s' if pluralize and count > 1 else ''} at {location}" for location, count in location_counts.items()]) + "."
            
        else:
            parts = []
            for y, start_x, end_x in sorted(lines):
                parts.append(f"{y} (cols {start_x}-{end_x})")
            # Fix unbound variable 'count'
            count = len(lines)
            location_description = f"at row{'s' if pluralize and count > 1 else ''} " + ", ".join(parts)
        
            plural = label + "s" if pluralize and count > 1 else label
            return f" {describe_quantity(count) if coarse_counts else count} {plural} " + location_description + "."

    else: # Not describing locations at all
        count = len(lines)
        return f" {describe_quantity(count) if coarse_counts else count} {label}{'s' if pluralize and count != 1 else ''}."

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
        for start_row in range(0 if verticality == 1 else 3, height - 2 if verticality == 1 else height):
            if is_staircase_from(scene, id_to_char, tile_descriptors, col, start_row, verticality, already_accounted):
                #print(f"staircase at {start_row} {col} {verticality}")
                # Now count how many columns this staircase extends
                length = 3
                while col + length < width and is_staircase_from(scene, id_to_char, tile_descriptors, col + length - 2, start_row + verticality*(length - 2), verticality, already_accounted):
                    length += 1
                staircases += 1
                col += length  # Skip past this staircase
                staircase_lengths.append(length)
                #print(f"staircase length {length} {already_accounted}")
                break  # Restart staircase search from new col
        else:
            col += 1  # No staircase starting here, move right

    type = "descending" if verticality == 1 else "ascending"
    if staircases > 0:
        return f" {describe_quantity(staircases) if coarse_counts else staircases} {type} staircase{'s' if pluralize and staircases > 1 else ''}"+ (f" with length{'s' if staircases > 1 else ''} {', '.join(map(str, staircase_lengths))}" if give_staircase_lengths else "")+ "."
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
            if row == len(scene): 
                return False # Do not count floor in staircases
            col = start_col + step
            tile = scene[row][col]
            if "solid" not in tile_descriptors.get(id_to_char[tile], []):
                #if start_col == 0: print("not solid at", row, col)
                return False
            # Check above this block is passable
            if row > 0:
                tile_above = scene[row - 1][col]
                if "solid" in tile_descriptors.get(id_to_char[tile_above], []):
                    #if start_col == 0: print("solid above", row, col)
                    return False
                else:
                    # Blocks beneath the stairs are also part of stairs
                    row2 = row
                    while row2 < len(scene) and "solid" in tile_descriptors.get(id_to_char[scene[row2][col]], []): 
                        blocks_in_stairs.add( (row2,col) )
                        row2 += 1                    

        # Only add all of the blocks once it is confirmed to be a staircase
        already_accounted.update(blocks_in_stairs)
        #if start_col == 0: print("staircase at", start_row, start_col, verticality)
        return True
    except IndexError:
        print(f"IndexError at start_col {start_col}, start_row {start_row}, verticality {verticality}")
        return False  # Out of bounds means no staircase


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
                if pipes or len(structure) >= 3:  # Ignore tiny groups of blocks, but keep all pipes
                    structures.append(structure)
                    already_accounted.update(structure)

    return structures

def valid_pipe(top_row, left_column, scene, char_to_id):
    """
        Is this a valid pipe or not?

        <>
        []
       ...
        []
    """
    # Case: left edge of screen
    if left_column == 0 and scene[top_row][left_column] == char_to_id['>']:
        # go down looking for ] or >
        row = top_row+1
        while row < len(scene):
            # I changed my mind on the emptiness check, but mainly because of bad data from SMB2. Might restore this check if I fix VGLC data
            if scene[row][left_column] in [char_to_id['<'], char_to_id['[']]: #, char_to_id['-']]: # emptiness under base also invalid
                return False
            elif scene[row][left_column] in [char_to_id['>'], char_to_id[']']]:
                row += 1
            else:
                return True

        return True
    # Case: right edge of screen
    elif left_column == len(scene[0]) - 1 and scene[top_row][left_column] == char_to_id['<']:
        # go down looking for [ or <
        row = top_row+1
        while row < len(scene):
            if scene[row][left_column] in [char_to_id['<'], char_to_id['[']]:
                row += 1
            # I changed my mind on the emptiness check, but mainly because of bad data from SMB2. Might restore this check if I fix VGLC data
            elif scene[row][left_column] in [char_to_id['>'], char_to_id[']']]: #, char_to_id['-']]:
                return False
            else:
                return True

        return True

    # Case: Full pipe
    elif left_column < len(scene[0]) - 1 and scene[top_row][left_column] == char_to_id['<'] and scene[top_row][left_column+1] == char_to_id['>']:
        # go down looking for [] or <>
        row = top_row+1
        while row < len(scene):
            if (scene[row][left_column] == char_to_id['<'] and scene[row][left_column+1] == char_to_id['>']) or (scene[row][left_column] == char_to_id['['] and scene[row][left_column+1] == char_to_id[']']):
                row += 1
            # I changed my mind on the emptiness check, but mainly because of bad data from SMB2. Might restore this check if I fix VGLC data
            elif scene[row][left_column] in [char_to_id['<'], char_to_id['['], char_to_id['>'], char_to_id[']']] or scene[row][left_column+1] in [char_to_id['<'], char_to_id['['], char_to_id['>'], char_to_id[']']]:
                return False
            else:
                return True

        return True

    return False

def valid_upside_down_pipe(bottom_row, left_column, scene, char_to_id):
    """
        Is this a valid upside down pipe or not?

        []
        []
        <>
    """
    # Case: left edge of screen
    if left_column == 0 and scene[bottom_row][left_column] == char_to_id['>']:
        # go up looking for ] or >
        row = bottom_row - 1
        while row >= 0:
            if scene[row][left_column] in [char_to_id['<'], char_to_id['['], char_to_id['-']]:  # emptiness above base also invalid
                return False
            elif scene[row][left_column] in [char_to_id['>'], char_to_id[']']]:
                row -= 1
            else:
                return True

        return True
    # Case: right edge of screen
    elif left_column == len(scene[0]) - 1 and scene[bottom_row][left_column] == char_to_id['<']:
        # go up looking for [ or <
        row = bottom_row - 1
        while row >= 0:
            if scene[row][left_column] in [char_to_id['<'], char_to_id['[']]:
                row -= 1
            elif scene[row][left_column] in [char_to_id['>'], char_to_id[']'], char_to_id['-']]:
                return False
            else:
                return True

        return True

    # Case: Full upside down pipe
    elif left_column < len(scene[0]) - 1 and scene[bottom_row][left_column] == char_to_id['<'] and scene[bottom_row][left_column + 1] == char_to_id['>']:
        # go up looking for [] or <>
        row = bottom_row - 1
        while row >= 0:
            if (scene[row][left_column] == char_to_id['['] and scene[row][left_column + 1] == char_to_id[']']) or (scene[row][left_column] == char_to_id['<'] and scene[row][left_column + 1] == char_to_id['>']):
                row -= 1
            elif scene[row][left_column] in [char_to_id['<'], char_to_id['['], char_to_id['>'], char_to_id[']'], char_to_id['-']] or scene[row][left_column + 1] in [char_to_id['<'], char_to_id['['], char_to_id['>'], char_to_id[']'], char_to_id['-']]:
                return False
            else:
                return True

        return True

    return False

def describe_structures(structures, ceiling_row=CEILING, floor_row=FLOOR, pipes=False, describe_absence=False, describe_locations=False, debug=False, scene=None, char_to_id=None, exclude_upside_down_pipes=False):
    """
        scene and char_to_id are needed when pipes is True so that the specific tiles can be checked.
        Returns a list of tuples (phrase, coordinates) where coordinates is a set of (row, col) positions
        associated with the phrase describing the structures of that type.
    """
    # Map each description to its list of structures
    desc_to_structs = {}
    
    for struct in structures:
        min_row = min(pos[0] for pos in struct)
        max_row = max(pos[0] for pos in struct)
        min_col = min(pos[1] for pos in struct)
        max_col = max(pos[1] for pos in struct)

        width = max_col - min_col + 1
        height = max_row - min_row + 1

        attached_to_ceiling = any(r == ceiling_row for r, c in struct)
        in_contact_with_floor = any(r == floor_row - 1 for r, c in struct)

        if pipes:
            if valid_pipe(min_row, min_col, scene, char_to_id):
                desc = "pipe"
            elif valid_upside_down_pipe(max_row, min_col, scene, char_to_id):
                if exclude_upside_down_pipes:
                    raise ValueError("Don't exclude_upside_down_pipes if valid upside down pipes are in the data")
                desc = "upside down pipe"
            else:
                desc = "broken pipe"
        else:
            if not attached_to_ceiling and width <= 2 and height >= 3 and in_contact_with_floor:
                desc = "tower"
            elif all((r, c) in struct for r in range(min_row, max_row + 1) for c in range(min_col, max_col + 1)):
                desc = "rectangular block cluster"
            #elif not attached_to_ceiling and width >= 3 and height <= 2 and in_contact_with_floor:
            #    desc = "wall"
            else:
                desc = "irregular block cluster"

        if debug:
            print(f"{desc} at {min_row} {max_row} {min_col} {max_col}: {struct}: attached_to_ceiling: {attached_to_ceiling}, in_contact_with_floor: {in_contact_with_floor}")

        if describe_locations:
            if coarse_locations:
                desc += " at " + describe_location((min_col + max_col) / 2.0, (min_row + max_row) / 2.0)
            else:
                desc += f" from row {min_row} to {max_row}, columns {min_col} to {max_col}"

        # Group structures by their description
        if desc not in desc_to_structs:
            desc_to_structs[desc] = []
        desc_to_structs[desc].append(struct)

    # Prepare phrases with their associated coordinates
    result = []
    
    # Process existing structures
    for desc, struct_list in desc_to_structs.items():
        count = len(struct_list)
        # Combine all coordinates for this description type
        all_coords = set()
        for struct in struct_list:
            all_coords.update(struct)
            
        if count == 1:
            # Need space in front
            phrase = f" one {desc}"
        else:
            # Pluralize the first word
            words = desc.split()
            for i in range(len(words)):
                if words[i] == "pipe":
                    words[i] = "pipes"
                elif words[i] == "tower":
                    words[i] = "towers"
                #elif words[i] == "wall":
                #    words[i] = "walls"
                elif words[i] == "cluster":
                    words[i] = "clusters"
            phrase = f" {describe_quantity(count)} " + " ".join(words)
        
        result.append((phrase + ".", all_coords))

    # Handle absence descriptions if needed
    if describe_absence:
        absent_types = {"pipe": set(), "upside down pipe" : set()} if pipes else {"tower": set(), "rectangular block cluster": set(), "irregular block cluster": set()}
        described_types = desc_to_structs.keys()
        
        for absent_type in absent_types:
            if absent_type not in described_types:
                if not (absent_type == "upside down pipe" and exclude_upside_down_pipes):
                    result.append((f" no {absent_type}s.", set()))

    return result if result else []

#def count_to_words(n):
#    words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
#    return words[n - 1] if 1 <= n <= 10 else str(n)

def generate_captions(dataset_path, tileset_path, output_path, describe_locations, describe_absence, exclude_upside_down_pipes=False, exclude_broken=True):
    """Processes the dataset and generates captions for each level scene."""
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    save_level_data(dataset, tileset_path, output_path, describe_locations, describe_absence, exclude_upside_down_pipes=exclude_upside_down_pipes, exclude_broken=exclude_broken)
    print(f"Captioned dataset saved to {output_path}")

def save_level_data(dataset, tileset_path, output_path, describe_locations, describe_absence, exclude_broken=True, exclude_upside_down_pipes=False, prompts=None):

    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)

    num_excluded = 0
    # Generate captions
    captioned_dataset = []
    for i, scene in enumerate(dataset):
        caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence, exclude_upside_down_pipes=exclude_upside_down_pipes)

        # We only want to discard levels with broken pipes if we indicate that (describe_absence is True)
        if exclude_broken and "broken" in caption:
            if "broken pipe" in caption:
                print("Broken pipe in training data")
            if "broken cannon" in caption:
                print("Broken cannon in training data")
            print(caption)
            current = len(captioned_dataset)
            print(f"Excluding training sample: {current}")
            num_excluded += 1
            
            #import torch
            #import torch.nn.functional as F
            #scene_tensor = torch.tensor(scene, dtype=torch.long)
            #one_hot_scene = F.one_hot(scene_tensor, num_classes=13).float() 
            #one_hot_scene = one_hot_scene.permute(2, 0, 1)
            #scene = one_hot_scene.unsqueeze(0)
            #from level_dataset import visualize_samples
            #image = visualize_samples(scene)
            #image.show()
            #if input("Press Enter to continue or type 'q' to quit: ") == 'q':
            #    print("Exiting caption generation.")
            #    sys.exit(0)

            continue

        captioned_dataset.append({
            "prompt": prompts[i] if prompts else None,
            "scene": scene,
            "caption": caption
        })

    # Probably need to fix the VGLC data manually.
    # Should I make the script repair the data or make my own fork of VGLC with good data?
    print(f"{num_excluded} samples excluded due to broken pipes/cannons.")

    # Save new dataset with captions
    with open(output_path, "w") as f:
        json.dump(captioned_dataset, f, indent=4)

def assign_caption(scene, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence, debug=False, return_details=False, exclude_upside_down_pipes=False):
    """Assigns a caption to a level scene based on its contents."""
    already_accounted = set()
    details = {} if return_details else None
    WIDTH = len(scene[0])

    # Include all of floor, even empty tiles
    for x in range(WIDTH):
        already_accounted.add((FLOOR, x))

    floor_row = FLOOR
    # Check if the row above the floor is identical to the floor row.
    # Some levels in SMB2 have a doubly thick floor.
    # There is also a special case when pipes are embedded in a thick floor. The pipe lip makes the
    # two rows unequal, but this is still an example of a double thick floor.
    if scene[FLOOR] == list(map(lambda x : char_to_id['['] if x == char_to_id['<'] else char_to_id[']'] if x == char_to_id['>'] else x, scene[FLOOR - 1])):
        floor_row = FLOOR - 1
        for x in range(WIDTH):
            already_accounted.add((FLOOR - 1, x))

    def add_to_caption(phrase, contributing_blocks):
        nonlocal caption
        #if phrase and "ceiling" in phrase:
        #    raise ValueError(f"{phrase} {contributing_blocks}")

        if phrase:
            caption += phrase
            if return_details and details is not None:
                details[phrase.strip()] = contributing_blocks

    caption = ""

    # Analyze floor
    floor_caption = analyze_floor(scene, id_to_char, tile_descriptors, describe_absence)
    add_to_caption(floor_caption + "." if floor_caption else "", list(already_accounted))

    def bigger_ceiling(ceiling_higher, ceiling_regular):
        if ceiling_higher == None:
            return False
        ceiling_order = ["full ceiling.", "ceiling with one gap.", "ceiling with two gaps.", "ceiling with a few gaps.", "ceiling with several gaps.", "ceiling with many gaps.", "no ceiling.", ""]
        return ceiling_order.index(ceiling_higher.strip()) <= ceiling_order.index(ceiling_regular.strip())

    # Analyze ceiling
    for c in range(CEILING, 0, -1):
        ceiling_regular = analyze_ceiling(scene, id_to_char, tile_descriptors, describe_absence, ceiling_row = c)
        ceiling_higher = analyze_ceiling(scene, id_to_char, tile_descriptors, describe_absence, ceiling_row = c - 1)
        ceiling_start = c
        #print(f"{c} ceiling_regular: {ceiling_regular}, ceiling_higher: {ceiling_higher}")
        if describe_absence and (ceiling_regular != " no ceiling." or ceiling_higher != " no ceiling."):
            break
        if not describe_absence and (ceiling_regular != "" or ceiling_higher != ""):
            break

    #print(f"END ceiling_regular: {ceiling_regular}, ceiling_higher: {ceiling_higher}")
        
    ceiling_phrase = None
    ceiling_row = None
    if (ceiling_regular == " no ceiling." and ceiling_higher == " no ceiling.") or (ceiling_regular == "" and ceiling_higher == ""):
        ceiling_row = None
        ceiling_phrase = ceiling_regular
        add_to_caption(ceiling_regular, []) # No ceiling at all
    elif ceiling_regular and ceiling_regular != " no ceiling." and ceiling_regular != "" and not bigger_ceiling(ceiling_higher, ceiling_regular):
        ceiling_row = ceiling_start
        ceiling_phrase = ceiling_regular
        add_to_caption(ceiling_regular, [(ceiling_start, x) for x in range(WIDTH)])
        for x in range(WIDTH):
            already_accounted.add((ceiling_start, x))
    elif ceiling_higher and ceiling_higher != " no ceiling." and ceiling_higher != "" and ceiling_start != 0:
        ceiling_row = ceiling_start - 1
        ceiling_phrase = ceiling_higher
        add_to_caption(ceiling_higher, [(ceiling_start - 1, x) for x in range(WIDTH)])
        for x in range(WIDTH):
            already_accounted.add((ceiling_start - 1, x))
    
    #print("after ceiling", (10,0) in already_accounted)
    
    # Is the ceiling filled in even more? (Some SML levels do this)
    if ceiling_row and ceiling_phrase:
        for r in range(ceiling_row - 1, -1, -1):
            #print(r ,f"also ceiling '{ceiling_phrase.strip()}'", details)
            if scene[r] == scene[ceiling_row]:
                if details:
                    details[ceiling_phrase.strip()].extend([(r, x) for x in range(WIDTH)])
                already_accounted.update((r, x) for x in range(WIDTH))
            
    # Count enemies
    enemy_phrase = count_caption_phrase(scene, [char_to_id['E']], "enemy", "enemies", describe_absence=describe_absence)
    add_to_caption(enemy_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t == char_to_id['E']])

    #print("after enemy", (10,0) in already_accounted)

    # Count question blocks
    question_block_phrase = count_caption_phrase(scene, [char_to_id['Q'], char_to_id['?']], "question block", "question blocks", describe_absence=describe_absence)
    add_to_caption(question_block_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in [char_to_id['Q'], char_to_id['?']]])

    #print("after qb", (10,0) in already_accounted)
    #print(already_accounted)
    # Count cannons
    cannon_phrase = count_caption_phrase(scene, [char_to_id['B']], "cannon", "cannons", describe_absence=describe_absence)
    cannon_locations = [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t == char_to_id['B']]
    add_to_caption(cannon_phrase, cannon_locations)
    already_accounted.update(cannon_locations)

    #print("after cannon", (10,0) in already_accounted)

    # Describe broken cannons
    broken_cannon_phrase = describe_broken_cannons(scene, char_to_id)
    add_to_caption(broken_cannon_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t == char_to_id['B']])

    # Count coins
    coin_phrase = count_caption_phrase(scene, [char_to_id['o']], "coin", "coins", describe_absence=describe_absence)
    add_to_caption(coin_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t == char_to_id['o']])

    # Coin lines
    coin_lines = find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor="coin", min_run_length=2)
    coin_line_phrase = describe_horizontal_lines(coin_lines, "coin line", describe_locations, describe_absence=describe_absence)
    add_to_caption(coin_line_phrase, [(y, x) for y, start_x, end_x in coin_lines for x in range(start_x, end_x + 1)])

    #print("after coin line", (10,0) in already_accounted)
    # Platforms
    platform_lines = find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor="solid", min_run_length=2, require_above_below_not_solid=True, already_accounted=already_accounted, exclude_rows=[] if ceiling_row == None else [ceiling_row])
    #print("after platform_lines", (10,0) in already_accounted)
    platform_phrase = describe_horizontal_lines(platform_lines, "platform", describe_locations, describe_absence=describe_absence)
    add_to_caption(platform_phrase, [(y, x) for y, start_x, end_x in platform_lines for x in range(start_x, end_x + 1)])

    #print("after platform", (10,0) in already_accounted)
    #print("before stairs", (10,0) in already_accounted)
    # Staircases
    up_stair_set = set()
    ascending_caption = analyze_staircases(scene, id_to_char, tile_descriptors, -1, already_accounted=up_stair_set)
    add_to_caption(ascending_caption, list(up_stair_set))
    #print(already_accounted)
    already_accounted.update(up_stair_set)

    down_stair_set = set()
    descending_caption = analyze_staircases(scene, id_to_char, tile_descriptors, 1, already_accounted=down_stair_set)
    add_to_caption(descending_caption, list(down_stair_set))
    #print(already_accounted)
    already_accounted.update(down_stair_set)

    #print(already_accounted)
    if describe_absence and not ascending_caption:
        add_to_caption(" no ascending staircases.", [])

    if describe_absence and not descending_caption:
        add_to_caption(" no descending staircases.", [])


    # Solid structures

    #print(already_accounted)
    pipe_set = set() # pipes can double count with floor, but there should be no other conflicts
    structures = find_solid_structures(scene, id_to_char, tile_descriptors, pipe_set, pipes=True)
    pipe_phrase = describe_structures(structures, pipes=True, describe_locations=describe_locations, describe_absence=describe_absence, debug=debug, scene=scene, char_to_id=char_to_id, exclude_upside_down_pipes=exclude_upside_down_pipes)
    for phrase, coords in pipe_phrase:
        add_to_caption(phrase, coords)
    
    already_accounted.update(pipe_set)

    #print(already_accounted)
    structures = find_solid_structures(scene, id_to_char, tile_descriptors, already_accounted)
    structure_phrase = describe_structures(structures, describe_locations=describe_locations, describe_absence=describe_absence, debug=debug, ceiling_row=ceiling_row, floor_row=floor_row)
    for phrase, coords in structure_phrase:
        add_to_caption(phrase, coords)

    #print(already_accounted)
    loose_block_phrase = count_caption_phrase(scene, [char_to_id['X'], char_to_id['S']], "loose block", "loose blocks", describe_absence=describe_absence, exclude=already_accounted)
    add_to_caption(loose_block_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in [char_to_id['X'], char_to_id['S']] and (r, c) not in already_accounted])

    return (caption.strip(), details) if return_details else caption.strip()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate captions for Mario screenshots")
    parser.add_argument("--dataset", required=True, help="json with level scenes")
    
    # Fix unsupported escape sequence in argument parser
    def escape_path(path):
        return path.replace("\\", "\\\\")

    parser.add_argument("--tileset", default=escape_path(common_settings.MARIO_TILESET), help="Descriptions of individual tile types")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    #parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--exclude_upside_down_pipes", action="store_true", default=False, help="Whether any mention of upside down pipes should be in captions")
    parser.add_argument("--include_broken", action='store_true', default=False, help="Whether any mention of upside down pipes should be in captions")
    global args
    args = parser.parse_args()

    exclude_broken = not args.include_broken

    dataset_file = args.dataset
    tileset_file = args.tileset
    output_file = args.output

    if not os.path.isfile(dataset_file) or not os.path.isfile(tileset_file):
        print("Error: One or more input files do not exist.")
        sys.exit(1)

    generate_captions(dataset_file, tileset_file, output_file, False, args.describe_absence, args.exclude_upside_down_pipes, exclude_broken=exclude_broken)