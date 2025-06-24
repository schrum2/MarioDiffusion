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
    parser.add_argument("--num_runs", type=int, default=1, help="The number of astar runs to do")



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
    #Do the full dataset instead of just 100 if we have more
    if(samples_json!=allsamples_json):
        create_astar_metrics(allsamples_json, args.output_dir, fullsamples=True, num_runs=args.num_runs)
    create_astar_metrics(samples_json, args.output_dir, num_runs=args.num_runs)



def create_astar_metrics(allsamples_json, output_dir, fullsamples=False, num_runs=1):
    # Load generated levels and real levels
    with open(allsamples_json, "r") as f:
        generated_data = json.load(f)


    # Annoyingly this doesn't get saved properly if we don't provide a path inside the folder, instead of the folder itself
    output_path = os.path.join(output_dir, "evaluation_metrics.json")
    generated_scenes = [entry for entry in generated_data if "scene" in entry and entry['scene']]

    savename="astar_result.jsonl"
    if fullsamples:
        savename = "full_"+savename

    metrics.astar_metrics(generated_scenes, output_json_path=output_path, save_name=savename, num_runs=num_runs)  
    

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

    avg_min_edit_distance_from_real, perfect_matches = None, None

    #We should only call edit distance from real if the scenes are 16 blocks across
    if len(generated_scenes[0][0])==len(real_scenes[0][0]):
        # Average min edit distance from generated to real levels
        avg_min_edit_distance_from_real, perfect_matches = metrics.average_min_edit_distance_from_real(generated_scenes, real_scenes)

    # Broken pipe calculations (as percent of scenes with broken pipes)
    broken_pipe_percent = metrics.analyze_broken_pipes(generated_data, as_instance_of_feature=False)

    # Broken pipe calculations (as percent of all pipes)
    broken_pipes_percentage_of_pipes = metrics.analyze_broken_pipes(generated_data, as_instance_of_feature=True)

    # Broken cannon calculations (as percent of scenes with broken cannons)
    broken_cannon_percent = metrics.analyze_broken_cannons(generated_data, as_instance_of_feature=False)

    # Broken cannon calculations (as percent of all cannons)
    broken_cannons_percentage_of_cannons = metrics.analyze_broken_cannons(generated_data, as_instance_of_feature=True)

    

    # Save results to a JSON file
    results = {
        "average_min_edit_distance": avg_min_edit_distance,
        "average_min_edit_distance_from_real": avg_min_edit_distance_from_real,
        "perfect_matches_to_real": perfect_matches,
        "broken_pipes_percentage_in_dataset": broken_pipe_percent,
        "broken_pipes_percentage_of_pipes": broken_pipes_percentage_of_pipes,
        "broken_cannons_percentage_in_dataset": broken_cannon_percent,
        "broken_cannons_percentage_of_cannons": broken_cannons_percentage_of_cannons
    }

    output_path = os.path.join(output_dir, "MarioGPT_metrics_summary.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Metrics summary saved to {output_path}")


def create_100_samples(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    total=len(data)

    #Case where we need to prune samples
    if total>100:
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
            print(f"Warning: file name {new_name} already exists. This file will automatically be deleted")
            os.remove(new_name)
        os.rename(test, new_name)
        return new_name

    #Assume we can do everything just fine with 96
    return json_path



if __name__ == "__main__":
    main()