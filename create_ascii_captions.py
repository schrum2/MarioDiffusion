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

        caption += count_caption_phrase(scene, [char_to_id['E']], "enemy", "enemies")
        caption += count_caption_phrase(scene, [char_to_id['Q'],char_to_id['?']], "question block", "question blocks")
  
        pipe_at_edge = 1 if in_column(scene, 0, char_to_id['>']) else 0
        caption += count_caption_phrase(scene, [char_to_id['<']], "pipe", "pipes", pipe_at_edge)

        caption += count_caption_phrase(scene, [char_to_id['o']], "coin", "coins")


        # TODO: Add more detailed captioning logic here.
        # Example: You could analyze enemy types, platform heights, pipes, etc.

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
