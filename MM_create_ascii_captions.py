import json
import sys
import os
from captions.util import extract_tileset, describe_quantity, count_caption_phrase, flood_fill

import util.common_settings as common_settings




# The floor is the last row of the scene (0-indexed)
FLOOR = common_settings.MEGAMAN_HEIGHT - 1
CEILING = common_settings.MEGAMAN_HEIGHT - 14 # 2

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
    - Can require non-solid space above and below (for platforms)
    - exclude_rows may not be needed because of the alread_accounted set
    Returns a list of (y, start_x, end_x) tuples
    """
    lines = []
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0

    #print((10,0) in already_accounted)

    for y in range(height):  # Skip FLOOR row
        
        if y in exclude_rows:
            continue # Could skip ceiling

        x = 0
        while x < width:
            tile_char = id_to_char[scene[y][x]]
            descriptors = tile_descriptors.get(tile_char, [])

            if target_descriptor not in descriptors:
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

                if target_descriptor in descriptors:
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


def find_solid_structures(scene, id_to_char, tile_descriptors, already_accounted):
    """Find unaccounted solid block structures"""
    visited = set()
    structures = []

    for row in range(len(scene)):
        for col in range(len(scene[0])):
            if (row, col) in visited or (row, col) in already_accounted:
                continue
            tile = scene[row][col]
            descriptors = tile_descriptors.get(id_to_char[tile], [])
            if "solid" in descriptors:
                structure = flood_fill(scene, visited, row, col, id_to_char, tile_descriptors, already_accounted)
                if len(structure) >= 3:  # Ignore tiny groups of blocks
                    structures.append(structure)
                    already_accounted.update(structure)

    return structures


def describe_structures(structures, ceiling_row=CEILING, floor_row=FLOOR, describe_absence=False, describe_locations=False, debug=False):
    """
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


def find_ladders(scene, ladder_ids, already_accounted=set(), describe_absence=False):
    """
    Finds vertical runs of ladder tiles and classifies each by where it connects
    within the playable area (excluding the 2-row ceiling and the floor row):
    top only, bottom only, both (full height), or neither (middle).
    """
    ladders = []  # list of (start_y, end_y, x)
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0

    ceiling_row = 2          # rows 0-1 are non-playable ceiling
    floor_row = height - 1   # last row is the floor
    playable_top = ceiling_row
    playable_bottom = floor_row 

    for x in range(width):
        y = 0
        while y < height:
            if scene[y][x] not in ladder_ids:
                y += 1
                continue

            possible_locations = set()
            run_start = y
            while y < height:
                if scene[y][x] in ladder_ids:
                    possible_locations.add((y, x))
                    y += 1
                else:
                    break
            already_accounted.update(possible_locations)
            run_end = y - 1
            ladders.append((run_start, run_end, x))

    if not ladders:
        if describe_absence:
            return [(" no ladders.", set())]
        else:
            return []

    categories = {"both": [], "top": [], "bottom": [], "middle": []}
    for start_y, end_y, x in ladders:
        connects_top = start_y <= playable_top
        connects_bottom = end_y >= playable_bottom
        if connects_top and connects_bottom:
            categories["both"].append((start_y, end_y, x))
        elif connects_top:
            categories["top"].append((start_y, end_y, x))
        elif connects_bottom:
            categories["bottom"].append((start_y, end_y, x))
        else:
            categories["middle"].append((start_y, end_y, x))

    suffix_map = {"top": "at top", "bottom": "at bottom", "middle": "in the middle"}

    # One (phrase, coords) tuple per ladder category, so each subtype gets its own
    # entry in `details` instead of being merged into one combo string.
    result = []
    for cat in ("both", "top", "bottom", "middle"):
        items = categories[cat]
        if not items:
            continue
        count = len(items)
        plural = "s" if pluralize and count != 1 else ""
        quantity = describe_quantity(count) if coarse_counts else count

        if cat == "both":
            phrase = f" {quantity} full height ladder{plural}."
        else:
            phrase = f" {quantity} ladder{plural} {suffix_map[cat]}."

        coords = {(y, x) for start_y, end_y, x in items for y in range(start_y, end_y + 1)}
        result.append((phrase, coords))

    return result

def find_water_caption(scene, empty_ids, water_ids, describe_absence=False):
    """
        Finds the ratio of water to all empty tiles, and returns the caption for it
    """
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0

    empty_count = 0
    water_count = 0

    for x in range(width):
        for y in range(height):
            id_at_loc = scene[y][x] # Get char at location

            if id_at_loc in empty_ids:
                empty_count += 1
                if id_at_loc in water_ids: #Water tiles should always be empty tiles as well, so we nest them to prevent errors
                    water_count += 1
    
    if empty_count==0 or water_count==0: #Need an escape so we don't devide by 0
        if describe_absence:
            return " no water."
        else:
            return ""
    
    ratio = water_count/empty_count
    
    if ratio < 0.35:
        return " some water."
    elif ratio >= .35 and ratio < .65:
        return " half water."
    elif ratio >= .65 and ratio != 1.0:
        return " mostly water."
    elif ratio == 1.0:
        return " all water."
    
    raise ValueError(f"It shouldn't be possible to get here. Error in describing water with air/water ratio of {ratio}")
    

# We need a seperate function so we avoid counting things like spikes as the ceiling
def analyze_ceiling(scene, wall_ids, describe_absence, ceiling_row = 2):
    """
    Analyzes ceiling row (0-based index) to detect a ceiling.
    Returns a caption phrase or an empty string if no ceiling is detected.
    """
    WIDTH = len(scene[0])

    row = scene[ceiling_row]
    #Count the number of solid tiles in the ceiling row
    solid_count = sum(1 for tile in row if tile in wall_ids)
    
    if solid_count == WIDTH:
        return " full ceiling."
    elif solid_count > WIDTH//2:
        # Count contiguous gaps of passable tiles
        gaps = 0
        in_gap = False
        for tile in row:
            # Get gaps if the tile at a point isn't a solid block
            if tile not in wall_ids:
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

# We need another seperate function, for the same reason
def analyze_floor(scene, wall_ids, describe_absence, floor_row=15, ladder_ids=None):
    """Analyzes the last row of the 16X16 scene and generates a floor description."""
    if ladder_ids is None:
        ladder_ids = []
    WIDTH = len(scene[0])
    last_row = scene[floor_row]  # The FLOOR row of the scene
    # A ladder tile at the floor isn't a fall-through gap, so treat it like solid floor for this purpose
    non_gap_ids = set(wall_ids) | set(ladder_ids)
    solid_count = sum(1 for tile in last_row if tile in non_gap_ids)
    passable_count = sum(1 for tile in last_row if tile not in non_gap_ids)

    if solid_count == WIDTH:
        return " full floor."
    elif passable_count == WIDTH:
        if describe_absence:
            return " no floor."
        else:
            return ""
    elif solid_count > passable_count:
        # Count contiguous groups of passable tiles
        gaps = 0
        in_gap = False
        for tile in last_row:
            # Enemies are also a gap since they immediately fall into the gap. Ladders are not gaps.
            if tile not in non_gap_ids:
                if not in_gap:
                    gaps += 1
                    in_gap = True
            else:
                in_gap = False
        return f" floor with {describe_quantity(gaps) if coarse_counts else gaps} gap" + ("s." if pluralize and gaps != 1 else ".")
    else:
        # Count contiguous groups of solid tiles
        chunks = 0
        in_chunk = False
        for tile in last_row:
            if tile in non_gap_ids:
                if not in_chunk:
                    chunks += 1
                    in_chunk = True
            else:
                in_chunk = False
        return f" giant gap with {describe_quantity(chunks) if coarse_counts else chunks} chunk"+("s" if pluralize and chunks != 1 else "")+" of floor."


def generate_captions(dataset_path, tileset_path, output_path, describe_locations, describe_absence,
                      caption_mode="legacy", caption_key="deterministic_captions"):
    """Processes the dataset and generates captions for each level scene."""
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    save_level_data(dataset, tileset_path, output_path, describe_locations, describe_absence,
                    caption_mode=caption_mode, caption_key=caption_key)
    print(f"Captioned dataset saved to {output_path}")

def save_level_data(dataset, tileset_path, output_path, describe_locations, describe_absence,
                    caption_mode="legacy", caption_key="deterministic_captions"):
    """Add a deterministic caption to every scene.

    "legacy" stores it in the "caption" field; "keyed" stores it as a one-element list under
    caption_key (default "deterministic_captions"), so a scene can carry captions from several
    sources at once. Either way every other input attribute is copied through, so passing a
    dataset that already carries metadata or LLM captions accumulates sources rather than
    replacing them.
    """

    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)

    # Generate captions
    captioned_dataset = []
    for i, combined_scene in enumerate(dataset):
        # Blank for Mega Man
        is_dict = isinstance(combined_scene, dict)
        if is_dict:
            scene = combined_scene['scene']
            data = combined_scene.get('data', None)
        else:
            scene = combined_scene
            data = None
        caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence, data)

        # Copy all input attributes (metadata + captions from other sources) so they carry
        # through; only the scene/caption fields below are (re)written.
        entry = dict(combined_scene) if is_dict else {}
        entry["scene"] = scene
        if caption_mode == "keyed":
            entry[caption_key] = [caption]
        else:
            entry["caption"] = caption
        captioned_dataset.append(entry)

    # Save new dataset with captions
    with open(output_path, "w") as f:
        json.dump(captioned_dataset, f, indent=4)

def detect_edge_walls(scene, wall_ids, ceiling_row=2, floor_row=15):
    """
    Detects 'left wall', 'perforated left wall', 'right wall', or 'perforated right wall'.
    
    1. Solid Wall: A SINGLE flood-filled component of solid blocks within the 
       3 outermost columns spans continuously from `ceiling_row` to `floor_row`.
    2. Perforated Wall: No single flood-filled shape spans top-to-bottom, but 
       there are NO contiguous vertical gaps >= 2 tiles tall across the 3 boundary columns.
    3. None: Any vertical gap >= 2 tiles tall exists across all 3 columns.
    """
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0
    wall_set = set(wall_ids)

    def evaluate_side(col_indices):
        valid_cols = set(col_indices)
        
        # Collect all solid blocks in the 3-column boundary strip within playable rows
        solid_nodes = {
            (r, c) for r in range(ceiling_row, floor_row + 1)
            for c in col_indices if scene[r][c] in wall_set
        }

        # --- Pass 1: Check for a SINGLE Flood-Filled Shape from Top to Bottom ---
        visited = set()
        for start_node in solid_nodes:
            if start_node in visited:
                continue

            # BFS / Flood Fill for a SINGLE component
            queue = [start_node]
            component = {start_node}
            visited.add(start_node)
            
            reaches_top = False
            reaches_bottom = False

            while queue:
                curr_r, curr_c = queue.pop(0)
                if curr_r == ceiling_row:
                    reaches_top = True
                if curr_r == floor_row:
                    reaches_bottom = True

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    neighbor = (nr, nc)
                    if (ceiling_row <= nr <= floor_row and 
                        nc in valid_cols and 
                        neighbor in solid_nodes and 
                        neighbor not in visited):
                        
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)

            # If this SINGLE connected component spans from top to bottom, it's a solid wall
            if reaches_top and reaches_bottom:
                return "solid", component

        # --- Pass 2: Perforated Wall Check ---
        # If no single connected shape spans top-to-bottom, evaluate vertical gaps row-by-row
        perforated_coords = set()
        row_has_solid = []

        for r in range(ceiling_row, floor_row + 1):
            has_solid = False
            for c in col_indices:
                if scene[r][c] in wall_set:
                    has_solid = True
                    perforated_coords.add((r, c))
            row_has_solid.append(has_solid)

        # Calculate max contiguous vertical gap (rows without any solid tile)
        max_gap = 0
        current_gap = 0
        for has_solid in row_has_solid:
            if not has_solid:
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0

        # Gaps <= 1 tile qualify as a perforated wall; gaps >= 2 tiles allow passage (no wall)
        if max_gap <= 1:
            return "perforated", perforated_coords

        return "none", set()

    left_cols = [0, 1, 2]
    right_cols = [width - 1, width - 2, width - 3]

    left_type, left_coords = evaluate_side(left_cols)
    right_type, right_coords = evaluate_side(right_cols)

    return left_type, left_coords, right_type, right_coords

def assign_caption(scene, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence, data=None, debug=False, return_details=False):
    """Assigns a caption to a level scene based on its contents."""
    already_accounted = set()
    details = {} if return_details else None
    ladder_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'climbable' in value]
    enemy_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'enemy' in value]
    powerup_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'powerup' in value]
    empty_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'empty' in value] #Used for water ratio calculation
    water_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'water' in value] #Used for water ratio calculation
    hazard_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'hazard' in value]
    moving_plat_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'moving' in value]
    wall_ids = [char_to_id[key] for key, value in tile_descriptors.items() if (('solid' in value) and ('penetrable' not in value) and ("hazard" not in value))]
    disappearing_ids = [char_to_id["A"]] if "A" in char_to_id else [] #There's nothing unique about the descriptors for disappearing blocks, so we just set it here
    #Ideas:
    #Walls for each size/exit directions
    #Some kind of data transfer telling us which way the level is moving
        #DONE Encode "enter:", "exit:", and "blocked:", all giving us a direction
    #Check for ladders, enemies, powerups, water/air, spikes, moving/disappearing blocks
        #DONE Ladders: count number of vertical strips
        #DONE enemies: same as mario, raw count
        #DONE powerups: same 
        #DONE water:a little, a lot, half, mostly, all: mesures water/air ratio, 0-10% water, 10-40%, 40-60%, 60-99%, 100% respectivly
        #DONE Spikes: a few:0-5, a lot:6+
        #DONE Moving platforms: one, two, several, for 1, 2, 3+ continuous horizantal platforms
        #DONE Disappearing blocks: a few: 0-3, a lot:4+
    #Base checks, mostly unchanged
        #DONE Platforms (slightly expand definition of a platform)
        #DONE Loose blocks (same as mario)    
    

    def add_to_caption(phrase, contributing_blocks):
        nonlocal caption
        #if phrase and "ceiling" in phrase:
        #    raise ValueError(f"{phrase} {contributing_blocks}")

        if phrase:
            caption += phrase
            if return_details and details is not None and contributing_blocks != None:
                details[phrase.strip()] = contributing_blocks

    caption = ""

    
    if data != None: #Some systems cant give us this extra data
        #Add captions from encoded data
        entrance_direction = data['entrance_direction']
        exit_direction = data['exit_direction']
        
        add_to_caption(f" entrance direction {entrance_direction.lower()}.", None)
        add_to_caption(f" exit direction {exit_direction.lower()}.", None)
    else:
        entrance_direction = None
        exit_direction = None


    # Count enemies
    enemy_phrase = count_caption_phrase(scene, enemy_ids, "enemy", "enemies", describe_absence=describe_absence)
    add_to_caption(enemy_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in enemy_ids])


    # Count powerups
    powerup_phrase = count_caption_phrase(scene, powerup_ids, "powerup", "powerups", describe_absence=describe_absence)
    add_to_caption(powerup_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in powerup_ids])

    # Count hazards
    hazard_phrase = count_caption_phrase(scene, hazard_ids, "hazard", "hazards", describe_absence=describe_absence)
    add_to_caption(hazard_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in hazard_ids])


    # Count disappearing blocks
    disappearing_phrase = count_caption_phrase(scene, disappearing_ids, "disappearing block", "disappearing blocks", describe_absence=describe_absence)
    add_to_caption(disappearing_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in disappearing_ids])

    #Count water
    water_phrase = find_water_caption(scene, empty_ids, water_ids, describe_absence)
    add_to_caption(water_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in water_ids])


    # Ceiling
    ceiling_row = None
    if (exit_direction is not None and exit_direction.lower() in ("left", "right")): # Only track ceiling if we're moving horizantally (exit_direction is stored as uppercase enum name, e.g. "RIGHT")
        ceiling_row = 2 #Define this here so we don't ignore platforms on row 2 later if we're moving vertically
        ceiling_tiles = [(ceiling_row, c) for c, t in enumerate(scene[ceiling_row]) if t in wall_ids]
        ceiling_phrase = analyze_ceiling(scene, wall_ids, describe_absence, ceiling_row=ceiling_row)
        already_accounted.update(ceiling_tiles)
        add_to_caption(ceiling_phrase, ceiling_tiles)

    # Floor
    floor_row = len(scene) - 1
    floor_tiles = [(floor_row, c) for c, t in enumerate(scene[floor_row]) if t in wall_ids]
    floor_phrase = analyze_floor(
        scene,
        wall_ids,
        describe_absence=describe_absence,
        floor_row=floor_row,
        ladder_ids=ladder_ids
    )
    already_accounted.update(floor_tiles)
    add_to_caption(floor_phrase, floor_tiles)

    # --- Edge Wall Detection ---
    # Assumes the ceiling is on row 2, which works for 16x16 scenes but not for larger ones.
    left_type, left_wall_coords, right_type, right_wall_coords = detect_edge_walls(
        scene,
        wall_ids,
        ceiling_row=ceiling_row if ceiling_row else 2, # TODO: Generalize
        floor_row=floor_row
    )

    # Format Left Wall Caption
    if left_type == "solid":
        add_to_caption(" left wall.", list(left_wall_coords))
    elif left_type == "perforated":
        add_to_caption(" perforated left wall.", list(left_wall_coords))
    elif describe_absence:
        add_to_caption(" no left wall.", [])

    # Format Right Wall Caption
    if right_type == "solid":
        add_to_caption(" right wall.", list(right_wall_coords))
    elif right_type == "perforated":
        add_to_caption(" perforated right wall.", list(right_wall_coords))
    elif describe_absence:
        add_to_caption(" no right wall.", [])

    # Platforms
    # Count moving platforms
    moving_plat_lines = find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor="moving", min_run_length=1, require_above_below_not_solid=True, already_accounted=already_accounted, exclude_rows=[ceiling_row, floor_row])
    moving_plat_phrase = describe_horizontal_lines(moving_plat_lines, "moving platform", describe_locations, describe_absence=describe_absence)
    add_to_caption(moving_plat_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in moving_plat_ids])

    #Count regular platforms
    platform_lines = find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor="solid", min_run_length=2, require_above_below_not_solid=True, already_accounted=already_accounted, exclude_rows=[ceiling_row, floor_row])
    #print("after platform_lines", (10,0) in already_accounted)
    platform_phrase = describe_horizontal_lines(platform_lines, "platform", describe_locations, describe_absence=describe_absence)
    add_to_caption(platform_phrase, [(y, x) for y, start_x, end_x in platform_lines for x in range(start_x, end_x + 1)])


    # Solid structures
    
    #Count ladders
    ladder_phrases = find_ladders(scene, ladder_ids, already_accounted, describe_absence)
    for phrase, coords in ladder_phrases:
        add_to_caption(phrase, coords)

    structures = find_solid_structures(scene, id_to_char, tile_descriptors, already_accounted)
    structure_phrase = describe_structures(structures, describe_locations=describe_locations, describe_absence=describe_absence, debug=debug)

    for phrase, coords in structure_phrase:
        add_to_caption(phrase, coords)

    #print(already_accounted)
    loose_block_phrase = count_caption_phrase(scene, wall_ids, "loose block", "loose blocks", describe_absence=describe_absence, exclude=already_accounted)
    add_to_caption(loose_block_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in wall_ids and (r, c) not in already_accounted])

    return (caption.strip(), details) if return_details else caption.strip()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate captions for Mega Man screenshots")
    parser.add_argument("--dataset", required=True, help="json with level scenes")
    
    parser.add_argument("--tileset", default=common_settings.MM_FULL_TILESET, help="Descriptions of individual tile types")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--caption-mode", choices=["legacy", "keyed"], default="legacy",
                        help="Output schema. 'legacy' (default) writes the single 'caption' field. 'keyed' writes the caption as a "
                             "one-element list under --caption-key, so a scene can carry captions from several sources at once. Both "
                             "modes copy all other input attributes (metadata and captions from other sources) to the output.")
    parser.add_argument("--caption-key", default="deterministic_captions",
                        help="Key to store the caption list under when --caption-mode keyed. Default: deterministic_captions")
    global args
    args = parser.parse_args()

    dataset_file = args.dataset
    tileset_file = args.tileset
    output_file = args.output

    if not os.path.isfile(dataset_file) or not os.path.isfile(tileset_file):
        print("Error: One or more input files do not exist.")
        sys.exit(1)

    generate_captions(dataset_file, tileset_file, output_file, False, args.describe_absence,
                      caption_mode=args.caption_mode, caption_key=args.caption_key)
