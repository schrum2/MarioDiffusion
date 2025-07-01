import json
import split_data as split_data
import argparse
from util.sampler import scene_to_ascii, CustomSimulator
from tqdm import tqdm
from create_ascii_captions import extract_tileset
import util.common_settings as common_settings


tileset_path = common_settings.MARIO_TILESET

# Ensure the tileset path exists

title_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for MarioGPT metrics")

    parser.add_argument("--generated_levels", type=str, default="datasets\\MarioGPT_LevelsAndCaptions-regular-long.json", help="The filepath of the LevelsAndCaptions format for MarioGPT generated data")
    parser.add_argument("--start_index", type=int, default=12, help="The start location of the astar testing")
    parser.add_argument("--only_play_unbeaten", action='store_true', help="If true, only play levels that have not been beaten before")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run for each level")
    parser.add_argument("--visualization", type=str, choices=["NES", "SNES"], default="NES", help="[NES|SNES] determines jar file that launches Mario evaluation")
    parser.add_argument("--agent", type=str, choices=["astar", "human"], default="astar", help="[astar|human] which agent plays the level")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.visualization == "NES":
        jar_file = "NESMarioEval.jar"
    elif args.visualization == "SNES":
        jar_file = "MarioEval.jar"
    else:
        raise ValueError("Invalid visualization type. Choose either 'NES' or 'SNES'.")

    # Load generated levels and real levels
    with open(args.generated_levels, "r") as f:
        if args.generated_levels.endswith('json'):
            generated_data = json.load(f)
        elif args.generated_levels.endswith('jsonl'):
            generated_data = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file format. Please provide a JSON or JSONL file.")

    generated_scenes = [entry for entry in generated_data if "scene" in entry and entry['scene']]
    
    #Find the levels at and after the chosen index
    generated_scenes=generated_scenes[args.start_index:]


    for idx, entry in enumerate(tqdm(generated_scenes, desc="A* metrics", unit="level")):
        if args.only_play_unbeaten and entry.get("averages", {}).get("beaten", 0.0) == 1.0:
            print(f"Skipping level {idx + args.start_index} as it has been beaten before.")
            continue
        elif "averages" in entry and "beaten" in entry["averages"]:
            print(f"Previous beaten score: {entry['averages']['beaten']}")

        scene = entry.get("scene")

        ascii_level = scene_to_ascii(scene, id_to_char, True)

        for t in range(args.num_trials):
            print(f"Trial {t + 1} for level {idx + args.start_index}")
            sim = CustomSimulator(ascii_level, jar_path=jar_file)
            if args.agent == "astar":
                output = sim.astar(render=True)
            elif args.agent == "human":
                output = sim.interactive()

            print(f"Level {idx + args.start_index}")
            print(output)

if __name__ == "__main__":
    main()