import util.metrics as metrics
import json
import split_data as split_data
import os
import argparse
from util.sampler import scene_to_ascii, CustomSimulator
from tqdm import tqdm
from create_ascii_captions import extract_tileset


tileset_path = '..\TheVGLC\Super Mario Bros\smb.json'

# Ensure the tileset path exists

title_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for MarioGPT metrics")

    parser.add_argument("--generated_levels", type=str, default="datasets\\MarioGPT_LevelsAndCaptions-regular-long.json", help="The filepath of the LevelsAndCaptions format for MarioGPT generated data")
    parser.add_argument("--start_index", type=int, default=12, help="The start location of the astar testing")

    return parser.parse_args()


def main():
    args = parse_args()
    # Load generated levels and real levels
    with open(args.generated_levels, "r") as f:
        generated_data = json.load(f)


    generated_scenes = [entry for entry in generated_data if "scene" in entry and entry['scene']]
    
    #Find the levels at and after the chosen index
    generated_scenes=generated_scenes[args.start_index:]


    for idx, entry in enumerate(tqdm(generated_scenes, desc="A* metrics", unit="level")):
        scene = entry.get("scene")

        ascii_level = scene_to_ascii(scene, id_to_char, True)

        sim = CustomSimulator(ascii_level)
        output = sim.astar(render=True)



if __name__ == "__main__":
    main()