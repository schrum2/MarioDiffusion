import json
import sys
import os
from collections import Counter

# The width of generated scenes may not be 16
#WIDTH = 16
# height is 22 for Lode Runner
HEIGHT = 22

# The floor is the last row of the scene (0-indexed)
FLOOR = HEIGHT - 1
CEILING = 1

# This is used for describing locations, but it doesn't work well
# standard width of Lode Runner scenes are 32
STANDARD_WIDTH = 32

LEFT = STANDARD_WIDTH // 3
RIGHT = STANDARD_WIDTH - LEFT

TOP = (FLOOR - CEILING) // 3 + CEILING
BOTTOM = FLOOR - ((FLOOR - CEILING) // 3)

# Could define these via the command line, but for now they are hardcoded
coarse_locations = True
coarse_counts = True
pluralize = True
give_staircase_lengths = False

def describe_size(count):
    if count <= 4: return "small"
    else: return "big"

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

def describe_quantity(count):
    if count == 0: return "no"
    elif count == 1: return "one"
    elif count == 2: return "two"
    elif count < 5: return "a few"
    elif count < 10: return "several"
    else: return "many"

def get_tile_descriptors(tileset):
    """Creates a mapping from tile character to its list of descriptors."""
    result = {char: set(attrs) for char, attrs in tileset["tiles"].items()}
    # Fake tiles. Should these contain anything? Note that code elsewhere expects everything to be passable or solid
    result["!"] = {"passable"}
    result["*"] = {"passable"}
    return result

def analyze_floor(scene, id_to_char, tile_descriptors, describe_absence):
    """Analyzes the last row of the 32x32 scene and generates a floor description."""
    WIDTH = len(scene[0])
    last_row = scene[-1]  # The FLOOR row of the scene
    solid_count = sum(1 for tile in last_row if "solid" in tile_descriptors.get(id_to_char[tile], []))
    passable_count = sum(1 for tile in last_row if "passable" in tile_descriptors.get(id_to_char[tile], []))

    if solid_count == WIDTH:
        return "full floor"
    elif passable_count == WIDTH:
        if describe_absence:
            return "no floor"
        else:
            return ""
    elif solid_count > passable_count:
        # Count contiguous groups of passable tiles
        gaps = 0
        in_gap = False
        for tile in last_row:
            # Enemies are also a gap since they immediately fall into the gap
            if "passable" in tile_descriptors.get(id_to_char[tile], []) or "enemy" in tile_descriptors.get(id_to_char[tile], []):
                if not in_gap:
                    gaps += 1
                    in_gap = True
            elif "solid" in tile_descriptors.get(id_to_char[tile], []):
                in_gap = False
            else:
                print("error")
                print(tile)
                print(id_to_char[tile])
                print(tile_descriptors)
                print(tile_descriptors.get(id_to_char[tile], []))
                raise ValueError("Every tile should be passable, solid, or enemy")
        return f"floor with {describe_quantity(gaps) if coarse_counts else gaps} gap" + ("s" if pluralize and gaps != 1 else "")
    else:
        # Count contiguous groups of solid tiles
        chunks = 0
        in_chunk = False
        for tile in last_row:
            if "solid" in tile_descriptors.get(id_to_char[tile], []):
                if not in_chunk:
                    chunks += 1
                    in_chunk = True
            elif "passable" in tile_descriptors.get(id_to_char[tile], []) or "enemy" in tile_descriptors.get(id_to_char[tile], []):
                in_chunk = False
            else:
                print("error")
                print(tile)
                print(tile_descriptors)
                print(tile_descriptors.get(tile, []))
                raise ValueError("Every tile should be either passable or solid")
        return f"giant gap with {describe_quantity(chunks) if coarse_counts else chunks} chunk"+("s" if pluralize and chunks != 1 else "")+" of floor"

def count_in_scene(scene, tiles, exclude=set()):
    """ counts standalone tiles, unless they are in the exclude set """
    count = 0
    for r, row in enumerate(scene):
        for c, t in enumerate(row): 
            #if exclude and t in tiles: print(r,c, exclude)
            if (r,c) not in exclude and t in tiles:
                #if exclude: print((r,t), exclude, (r,t) in exclude)
                count += 1
    #if exclude: print(tiles, exclude, count)
    return count

def count_caption_phrase(scene, tiles, name, names, offset = 0, describe_absence=False, exclude=set()):
    """ offset modifies count used in caption """
    count = offset + count_in_scene(scene, tiles, exclude)
    #if name == "loose block": print("count", count)
    if count > 0: 
        return f" {describe_quantity(count) if coarse_counts else count} " + (names if pluralize and count > 1 else name) + "."
    elif describe_absence:
        return f" no {names}."
    else:
        return ""

def in_column(scene, x, tile):
    for row in scene:
        if row[x] == tile:
            return True

    return False

def analyze_ceiling(scene, id_to_char, tile_descriptors, describe_absence, ceiling_row = CEILING):
    """
    Analyzes ceiling row (0-based index) to detect a ceiling.
    Returns a caption phrase or an empty string if no ceiling is detected.
    """
    WIDTH = len(scene[0])

    row = scene[ceiling_row]
    solid_count = sum(1 for tile in row if "solid" in tile_descriptors.get(id_to_char[tile], []))
    
    if solid_count == WIDTH:
        return " full ceiling."
    elif solid_count > WIDTH//2:
        # Count contiguous gaps of passable tiles
        gaps = 0
        in_gap = False
        for tile in row:
            # Enemies are also a gap since they immediately fall into the gap, but they are marked as "moving" and not "passable"
            if "passable" in tile_descriptors.get(id_to_char[tile], []) or "moving" in tile_descriptors.get(id_to_char[tile], []):
                if not in_gap:
                    gaps += 1
                    in_gap = True
            else:
                in_gap = False
        result = f" ceiling with {describe_quantity(gaps) if coarse_counts else gaps} gap" + ("s" if pluralize and gaps != 1 else "") + "."

        # Adding the "moving" check should make this code unnecessary
        #if result == ' ceiling with no gaps.':
        #    print("This should not happen: ceiling with no gaps")
        #    print("ceiling_row:", scene[ceiling_row])
        #    result = " full ceiling."

        return result
    elif describe_absence:
        return " no ceiling."
    else:
        return ""  # Not enough solid tiles for a ceiling

def find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor, min_run_length=2, require_above_below_not_solid=False, exclude_rows = [], already_accounted = set()):
    """
    Finds horizontal lines (runs) of tiles with the target descriptor.
    - Skips the FLOOR row
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
            descriptors = tile_descriptors.get(tile_char, set())  # Ensure this is always a set

            # If required, check for passable tiles above and below
            if require_above_below_not_solid:
                # Above
                if y > 0:
                    above_char = id_to_char[scene[y - 1][x]]
                    if "solid" in tile_descriptors.get(above_char, set()):
                        x += 1
                        continue
                else:
                    x += 1
                    continue
                # Below
                if y + 1 < height:
                    below_char = id_to_char[scene[y + 1][x]]
                    if "solid" in tile_descriptors.get(below_char, set()):
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
                descriptors = tile_descriptors.get(tile_char, set())

                if (target_descriptor in descriptors):
                    if require_above_below_not_solid:
                        if y > 0 and "solid" in tile_descriptors.get(id_to_char[scene[y - 1][x]], set()):
                            break
                        if y + 1 < height and "solid" in tile_descriptors.get(id_to_char[scene[y + 1][x]], set()):
                            break

                    possible_locations.add( (y,x) )
                    x += 1
                else:
                    break
            run_length = x - run_start
            if run_length >= min_run_length:
                already_accounted.update(possible_locations) # Blocks of the line are now accounted for
                lines.append((y, run_start, x - 1))
            else:
                x += 1

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


def flood_fill(scene, visited, start_row, start_col, id_to_char, tile_descriptors, excluded, pipes=False):
    stack = [(start_row, start_col)]
    structure = []

    while stack:
        row, col = stack.pop()
        if (row, col) in visited or (row, col) in excluded:
            continue
        tile = scene[row][col]
        descriptors = tile_descriptors.get(id_to_char[tile], [])

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
                if structure:
                    structures.append(structure)

    return structures

def describe_structures(structures, ceiling_row=CEILING, floor_row=FLOOR, describe_absence=False, describe_locations=False, debug=False, scene=None, char_to_id=None):
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
                if words[i] == "tower":
                    words[i] = "towers"
                #elif words[i] == "wall":
                #    words[i] = "walls"
                elif words[i] == "cluster":
                    words[i] = "clusters"
            phrase = f" {describe_quantity(count)} " + " ".join(words)
        
        result.append((phrase + ".", all_coords))

    # Handle absence descriptions if needed
    if describe_absence:
        absent_types = {"tower": set(), "rectangular block cluster": set(), "irregular block cluster": set()}
        described_types = desc_to_structs.keys()
        
        for absent_type in absent_types:
            if absent_type not in described_types:
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

def extract_tileset(tileset_path):
    # Load tileset
    with open(tileset_path, "r") as f:
        tileset = json.load(f)
        #print(f"tileset: {tileset}")
        tile_chars = sorted(tileset['tiles'].keys())
        # I've been saying everywhere that the number of tiles is 10, but there are really only
        # 8 types. I can't remember why I wanted the wiggle room, but I think I should keep it for
        # now. However, this requires me to add some bogus tiles to the list.
        tile_chars.append('!') 
        tile_chars.append('*') 
        #print(f"tile_chars: {tile_chars}")
        id_to_char = {idx: char for idx, char in enumerate(tile_chars)}
        #print(f"id_to_char: {id_to_char}")
        char_to_id = {char: idx for idx, char in enumerate(tile_chars)}
        #print(f"char_to_id: {char_to_id}")
        tile_descriptors = get_tile_descriptors(tileset)
        #print(f"tile_descriptors: {tile_descriptors}")

    return tile_chars, id_to_char, char_to_id, tile_descriptors

def save_level_data(dataset, tileset_path, output_path, describe_locations, describe_absence):

    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)

    num_excluded = 0
    # Generate captions
    captioned_dataset = []
    for scene in dataset:
        # caption blank/empty for lode runner
        #caption = " "
        caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence)

        if "broken" in caption:
            print("Broken pipe in training data")
            print(caption)
            current = len(captioned_dataset)
            print(f"Excluding training sample: {current}")
            num_excluded += 1
            continue

        captioned_dataset.append({
            "scene": scene,
            "caption": caption
        })

    # Probably need to fix the VGLC data manually.
    # Should I make the script repair the data or make my own fork of VGLC with good data?
    print(f"{num_excluded} samples excluded due to broken pipes")

    # Save new dataset with captions
    with open(output_path, "w") as f:
        json.dump(captioned_dataset, f, indent=4)

def assign_caption(scene, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence, debug=False, return_details=False):
    """Assigns a caption to a level scene based on its contents."""
    already_accounted = set()
    details = {} if return_details else None
    WIDTH = len(scene[0])

    # Include all of floor, even empty tiles
    for x in range(WIDTH):
        already_accounted.add((FLOOR, x))

    floor_row = FLOOR

    def add_to_caption(phrase, contributing_blocks):
        nonlocal caption
        #if phrase and "ceiling" in phrase:
        #    raise ValueError(f"{phrase} {contributing_blocks}")

        if phrase:
            caption += phrase
            if return_details and details is not None:
                details[phrase.strip()] = contributing_blocks

    # blank for lode runner
    caption = ""
    #caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence)

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
    
    # Is the ceiling filled in even more? 
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

    # Count gold
    if 'G' in char_to_id:
        gold_phrase = count_caption_phrase(scene, [char_to_id['G']], "gold", "gold", describe_absence=describe_absence)
        add_to_caption(gold_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t == char_to_id['G']])

     # Count ropes
    if '-' in char_to_id:
        rope_lines = find_horizontal_lines(
            scene, id_to_char, tile_descriptors, target_descriptor="rope", min_run_length=1
        )
        rope_phrase = describe_horizontal_lines(rope_lines, "rope", describe_locations, describe_absence=describe_absence)
        add_to_caption(rope_phrase, [(y, x) for y, start_x, end_x in rope_lines for x in range(start_x, end_x + 1)])

    # Count ladders
    if '#' in char_to_id:
        ladder_lines = find_vertical_lines(
            scene, id_to_char, tile_descriptors, target_descriptor="ladder", min_run_length=1
        )
        ladder_phrase = describe_vertical_lines(ladder_lines, "ladder", describe_locations, describe_absence=describe_absence)
        add_to_caption(ladder_phrase, [(y, x) for x, start_y, end_y in ladder_lines for y in range(start_y, end_y + 1)])

    # Count player spawn (M) - only one allowed
    if 'M' in char_to_id:
        spawn_positions = [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t == char_to_id['M']]
        if len(spawn_positions) == 1:
            spawn_phrase = " one spawn point."
        elif len(spawn_positions) == 0 and describe_absence:
            spawn_phrase = " no spawn point."
        elif len(spawn_positions) > 1:
            spawn_phrase = f" {describe_quantity(len(spawn_positions)) if coarse_counts else len(spawn_positions)} spawn points."
        else:
            spawn_phrase = ""
        # Don't need to say there is a spawn point since there is only one
        #add_to_caption(spawn_phrase, spawn_positions)

    # Platforms
    platform_lines = find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor="solid", min_run_length=2, require_above_below_not_solid=True, already_accounted=already_accounted, exclude_rows=[] if ceiling_row == None else [ceiling_row])
    #print("after platform_lines", (10,0) in already_accounted)
    platform_phrase = describe_horizontal_lines(platform_lines, "platform", describe_locations, describe_absence=describe_absence)
    add_to_caption(platform_phrase, [(y, x) for y, start_x, end_x in platform_lines for x in range(start_x, end_x + 1)])


    # Solid structures

    #print(already_accounted)
    structures = find_solid_structures(scene, id_to_char, tile_descriptors, already_accounted)
    structure_phrase = describe_structures(structures, describe_locations=describe_locations, describe_absence=describe_absence, debug=debug, ceiling_row=ceiling_row, floor_row=floor_row)
    for phrase, coords in structure_phrase:
        add_to_caption(phrase, coords)

    #print(already_accounted)
    loose_block_phrase = count_caption_phrase(scene, [char_to_id['B'], char_to_id['b']], "loose block", "loose blocks", describe_absence=describe_absence, exclude=already_accounted)
    add_to_caption(loose_block_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in [char_to_id['B'], char_to_id['b']] and (r, c) not in already_accounted])

    return (caption.strip(), details) if return_details else caption.strip()

def find_vertical_lines(scene, id_to_char, tile_descriptors, target_descriptor, min_run_length=2, require_left_right_not_solid=False, exclude_cols = [], already_accounted = set()):
    """
    Finds vertical lines (runs) of tiles with the target descriptor.
    Used for ladders mainly.
    Returns a list of (x, start_y, end_y) tuples
    """
    lines = []
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0

    for x in range(width):
        if x in exclude_cols:
            continue
        y = 0
        while y < height:
            tile_char = id_to_char[scene[y][x]]
            descriptors = tile_descriptors.get(tile_char, set())

            # If required, check for passable tiles left and right
            if require_left_right_not_solid:
                # Left
                if x > 0:
                    left_char = id_to_char[scene[y][x - 1]]
                    if "solid" in tile_descriptors.get(left_char, set()):
                        y += 1
                        continue
                else:
                    y += 1
                    continue
                # Right
                if x + 1 < width:
                    right_char = id_to_char[scene[y][x + 1]]
                    if "solid" in tile_descriptors.get(right_char, set()):
                        y += 1
                        continue
                else:
                    y += 1
                    continue

            # Start of valid run
            possible_locations = set()
            run_start = y
            while y < height:
                tile_char = id_to_char[scene[y][x]]
                descriptors = tile_descriptors.get(tile_char, set())
                if (target_descriptor in descriptors):
                    if require_left_right_not_solid:
                        if x > 0 and "solid" in tile_descriptors.get(id_to_char[scene[y][x - 1]], set()):
                            break
                        if x + 1 < width and "solid" in tile_descriptors.get(id_to_char[scene[y][x + 1]], set()):
                            break
                    possible_locations.add((y, x))
                    y += 1
                else:
                    break
            run_length = y - run_start
            if run_length >= min_run_length:
                already_accounted.update(possible_locations)
                lines.append((x, run_start, y - 1))
            else:
                y += 1
    return lines

def describe_vertical_lines(lines, label, describe_locations, describe_absence):
    if not lines:
        if describe_absence:
            return f" no {label}s."
        else:
            return ""
    if describe_locations:
        if coarse_locations:
            location_counts = {}
            for x, start_y, end_y in sorted(lines):
                location_str = f"{describe_location(x, (end_y + start_y)/2.0)}"
                if location_str in location_counts:
                    location_counts[location_str] += 1
                else:
                    location_counts[location_str] = 1
            return " " + ". ".join([f"{describe_quantity(count) if coarse_counts else count} {label}{'s' if pluralize and count > 1 else ''} at {location}" for location, count in location_counts.items()]) + "."
        else:
            parts = []
            for x, start_y, end_y in sorted(lines):
                parts.append(f"{x} (rows {start_y}-{end_y})")
            count = len(lines)
            location_description = f"at col{'s' if pluralize and count > 1 else ''} " + ", ".join(parts)
            plural = label + "s" if pluralize and count > 1 else label
            return f" {describe_quantity(count) if coarse_counts else count} {plural} " + location_description + "."
    else:
        count = len(lines)
        return f" {describe_quantity(count) if coarse_counts else count} {label}{'s' if pluralize and count != 1 else ''}."


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate captions for Lode Runner screenshots")
    parser.add_argument("--dataset", required=True, help="json with level scenes")
    
    # Fix unsupported escape sequence in argument parser
    def escape_path(path):
        return path.replace("\\", "\\\\")

    parser.add_argument("--tileset", default=escape_path('..\\TheVGLC\\Lode Runner\\Loderunner.json'), help="Descriptions of individual tile types")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument('--target_height', type=int, default=32, help='Target output height (e.g., 32)')
    parser.add_argument('--target_width', type=int, default=32, help='Target output width (e.g., 32)')
    #parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    global args
    args = parser.parse_args()

    dataset_file = args.dataset
    tileset_file = args.tileset
    output_file = args.output

    if not os.path.isfile(dataset_file) or not os.path.isfile(tileset_file):
        print("Error: One or more input files do not exist.")
        sys.exit(1)

    generate_captions(dataset_file, tileset_file, output_file, False, args.describe_absence)
