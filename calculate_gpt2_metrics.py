import util.metrics as metrics
import json
import split_data as split_data
import os

def main():

    # Paths to the JSON files
    generated_file_path = "datasets\\MarioGPT_LevelsAndCaptions-100-samples.json"
    game_levels_file_path = "datasets\\Mar1and2_LevelsAndCaptions-regular.json"

    #average min edit distance with respect to generated levels themselves (average_min_edit_distance)
    #average min edit distance from real levels (average_min_edit_distance_from_real)
    #broken pipe calculations
    #broken cannon calculations
    #save this to a json file
    
    # Load generated levels and real levels
    with open(generated_file_path, "r") as f:
        generated_data = json.load(f)
    with open(game_levels_file_path, "r") as f:
        real_data = json.load(f)

    # Extract scenes (assumes each entry has a "scene" key)
    generated_scenes = [entry["scene"] for entry in generated_data if "scene" in entry and entry['scene']]
    real_scenes = [entry["scene"] for entry in real_data if "scene" in entry and entry['scene']]

    # 1. Average min edit distance among generated levels
    avg_min_edit_distance = metrics.average_min_edit_distance(generated_scenes)

    # 2. Average min edit distance from generated to real levels
    avg_min_edit_distance_from_real, perfect_matches = metrics.average_min_edit_distance_from_real(generated_scenes, real_scenes)

    # 3. Broken pipe calculations (as percent of scenes with broken pipes)
    broken_pipe_percent = metrics.analyze_broken_pipes(generated_data, as_instance_of_feature=False)

    # 4. Broken cannon calculations (as percent of scenes with broken cannons)
    broken_cannon_percent = metrics.analyze_broken_cannons(generated_data, as_instance_of_feature=False)

    astar_scenes = [entry for entry in generated_data if "scene" in entry and entry['scene']]
    metrics.astar_metrics(astar_scenes)

    # 5. Save results to a JSON file
    results = {
        "average_min_edit_distance_generated": avg_min_edit_distance,
        "average_min_edit_distance_from_real": avg_min_edit_distance_from_real,
        "perfect_matches_to_real": perfect_matches,
        "broken_pipe_percent": broken_pipe_percent,
        "broken_cannon_percent": broken_cannon_percent
    }

    output_path = "metrics_summary.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Metrics summary saved to {output_path}")
    print(json.dumps(results, indent=2))


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
    os.rename(new_name)

    return new_name


if __name__ == "__main__":
    main()