import util.metrics as metrics
import json
import split_data as split_data
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for MarioGPT metrics")

    parser.add_argument("--generated_levels", type=str, default="datasets\\MarioGPT_LevelsAndCaptions-regular.json", help="The filepath of the LevelsAndCaptions format for MarioGPT generated data")
    parser.add_argument("--training_levels", type=str, default="datasets\\Mar1and2_LevelsAndCaptions-regular.json", help="The filepath of the LevelsAndCaptions format for default data")
    parser.add_argument("--output_dir", type=str, default="MarioGPT_metrics", help="The output directory for the created json data")


    return parser.parse_args()

def main():

    args = parse_args()

    if os.path.exists(args.output_dir):
        print("Exiting. Please remove the directory or choose a different output directory.")
        exit()
    else:
        os.makedirs(args.output_dir)

    # Paths to the JSON files
    allsamples_json = args.generated_levels
    real_json=args.training_levels

    print("Creating 100 examples")
    samples_json=create_100_samples(allsamples_json)

    print("Finding basic metrics")
    find_metrics_of_samples(samples_json, real_json, args.output_dir)
    
    print("Finding astar metrics")
    create_astar_metrics(allsamples_json, args.output_dir)



def create_astar_metrics(allsamples_json, output_dir):
    # Load generated levels and real levels
    with open(allsamples_json, "r") as f:
        generated_data = json.load(f)

    generated_scenes = [entry for entry in generated_data if "scene" in entry and entry['scene']]

    metrics.astar_metrics(generated_scenes, output_json_path=output_dir, save_name="MarioGPT_astar_metrics.jsonl")
    print(f"Astar metrics saved to {output_dir}")


def find_metrics_of_samples(samples_json, real_json, output_dir):
    # Load generated levels and real levels
    with open(samples_json, "r") as f:
        generated_data = json.load(f)
    with open(real_json, "r") as f:
        real_data = json.load(f)

    # Extract scenes (assumes each entry has a "scene" key)
    generated_scenes = [entry["scene"] for entry in generated_data if "scene" in entry and entry['scene']]
    real_scenes = [entry["scene"] for entry in real_data if "scene" in entry and entry['scene']]

    # Average min edit distance among generated levels
    avg_min_edit_distance = metrics.average_min_edit_distance(generated_scenes)

    # Average min edit distance from generated to real levels
    avg_min_edit_distance_from_real, perfect_matches = metrics.average_min_edit_distance_from_real(generated_scenes, real_scenes)

    # Broken pipe calculations (as percent of scenes with broken pipes)
    broken_pipe_percent = metrics.analyze_broken_pipes(generated_data, as_instance_of_feature=False)

    # Broken cannon calculations (as percent of scenes with broken cannons)
    broken_cannon_percent = metrics.analyze_broken_cannons(generated_data, as_instance_of_feature=False)

    

    # Save results to a JSON file
    results = {
        "average_min_edit_distance_generated": avg_min_edit_distance,
        "average_min_edit_distance_from_real": avg_min_edit_distance_from_real,
        "perfect_matches_to_real": perfect_matches,
        "broken_pipe_percent": broken_pipe_percent,
        "broken_cannon_percent": broken_cannon_percent
    }

    output_path = os.path.join(output_dir, "MarioGPT_metrics_summary.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Metrics summary saved to {output_path}")


def create_100_samples(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    total=len(data)
    percentage_for_100=100/total
    remainder=.9-percentage_for_100

    #Split the dataset to find 100 samples
    train, val, test = split_data.split_dataset(json_path, remainder, .1, percentage_for_100)

    #remove json files for data we don't care about, not 100 samples
    os.remove(train)
    os.remove(val)

    directory = os.path.dirname(test)
    new_name=os.path.join(directory, "MarioGPT_LevelsAndCaptions-100-samples.json")
    if os.path.exists(new_name):
        print(f"Warning: file name {new_name} already exists. Loading previously made file for metrics creation, this may lead to unexpected behavior or results. Delete this file to ensure metrics work as expected.")
        os.remove(test)
    else:
        os.rename(test, new_name)

    return new_name


if __name__ == "__main__":
    main()