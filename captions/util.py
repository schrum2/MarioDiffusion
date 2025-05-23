import json
import sys
import os
from collections import Counter

# This file contains utility functions for analyzing and describing levels in both Lode Runner and Super Mario Bros.

# Could define these via the command line, but for now they are hardcoded
coarse_locations = True
coarse_counts = True
pluralize = True
give_staircase_lengths = False

def describe_size(count):
    if count <= 4: return "small"
    else: return "big"

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

def analyze_ceiling(scene, id_to_char, tile_descriptors, describe_absence, ceiling_row = 1):
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

def extract_tileset(tileset_path):
    # Load tileset
    with open(tileset_path, "r") as f:
        tileset = json.load(f)
        #print(f"tileset: {tileset}")
        tile_chars = sorted(tileset['tiles'].keys())
        # Wiggle room for the tileset to be a bit more flexible.
        # However, this requires me to add some bogus tiles to the list.
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
