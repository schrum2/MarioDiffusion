import json
import sys
import os

def get_tile_descriptors(tileset):
    """Creates a mapping from tile character to its list of descriptors."""
    return {char: set(attrs) for char, attrs in tileset["tiles"].items()}

def analyze_floor(scene, tile_descriptors):
    """Analyzes the last row of the 16x16 scene and generates a floor description."""
    last_row = scene[-1]  # The bottom row of the scene
    solid_count = sum(1 for tile in last_row if "solid" in tile_descriptors.get(tile, []))
    passable_count = sum(1 for tile in last_row if "passable" in tile_descriptors.get(tile, []))

    if solid_count == 16:
        return "full floor"
    elif passable_count == 16:
        return "no floor"
    else:
        # Count contiguous groups of passable tiles
        gaps = 0
        in_gap = False
        for tile in last_row:
            if "passable" in tile_descriptors.get(tile, []):
                if not in_gap:
                    gaps += 1
                    in_gap = True
            else:
                in_gap = False
        return f"floor with {gaps} gaps"

def generate_captions(dataset_path, tileset_path, output_path):
    """Processes the dataset and generates captions for each level scene."""
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Load tileset
    with open(tileset_path, "r") as f:
        tileset = json.load(f)
        tile_descriptors = get_tile_descriptors(tileset)

    # Generate captions
    captioned_dataset = []
    for scene in dataset:
        caption = analyze_floor(scene, tile_descriptors)

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
