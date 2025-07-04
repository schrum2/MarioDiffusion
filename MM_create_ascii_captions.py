import json
import sys
import os
from collections import Counter
from captions.util import extract_tileset, describe_size, describe_quantity, get_tile_descriptors, analyze_floor, count_in_scene, count_caption_phrase, in_column, analyze_ceiling, flood_fill

import util.common_settings as common_settings

# The floor is the last row of the scene (0-indexed)
FLOOR = common_settings.MEGAMAN_HEIGHT - 1
CEILING = common_settings.MEGAMAN_HEIGHT - 12 #  4

# This is used for describing locations, but it doesn't work well
STANDARD_WIDTH = common_settings.MEGAMAN_WIDTH

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

def generate_captions(dataset_path, tileset_path, output_path, describe_locations, describe_absence):
    """Processes the dataset and generates captions for each level scene."""
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    save_level_data(dataset, tileset_path, output_path, describe_locations, describe_absence)
    print(f"Captioned dataset saved to {output_path}")

def save_level_data(dataset, tileset_path, output_path, describe_locations, describe_absence):

    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)

    num_excluded = 0
    # Generate captions
    captioned_dataset = []
    for i, scene in enumerate(dataset):
        # Blank for Mega Man
        caption = ""
        #caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence, exclude_upside_down_pipes=exclude_upside_down_pipes)

        # We only want to discard levels with broken pipes if we indicate that (describe_absence is True)
            
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

        captioned_dataset.append({
            "scene": scene,
            "caption": caption
        })

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


    # Platforms
    platform_lines = find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor="solid", min_run_length=2, require_above_below_not_solid=True, already_accounted=already_accounted, exclude_rows=[] if ceiling_row == None else [ceiling_row])
    #print("after platform_lines", (10,0) in already_accounted)
    platform_phrase = describe_horizontal_lines(platform_lines, "platform", describe_locations, describe_absence=describe_absence)
    add_to_caption(platform_phrase, [(y, x) for y, start_x, end_x in platform_lines for x in range(start_x, end_x + 1)])


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
    parser = argparse.ArgumentParser(description="Generate captions for Mega Man screenshots")
    parser.add_argument("--dataset", required=True, help="json with level scenes")
    
    # Fix unsupported escape sequence in argument parser
    def escape_path(path):
        return path.replace("\\", "\\\\")

    parser.add_argument("--tileset", default=escape_path('..\\TheVGLC\\MegaMan\\MM.json'), help="Descriptions of individual tile types")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    #parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--exclude_upside_down_pipes", action="store_true", default=False, help="Whether any mention of upside down pipes should be in captions")
    global args
    args = parser.parse_args()

    dataset_file = args.dataset
    tileset_file = args.tileset
    output_file = args.output

    if not os.path.isfile(dataset_file) or not os.path.isfile(tileset_file):
        print("Error: One or more input files do not exist.")
        sys.exit(1)

    generate_captions(dataset_file, tileset_file, output_file, False, args.describe_absence)
