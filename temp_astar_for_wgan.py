import os
import json
from util.metrics import astar_metrics

root_dir = os.getcwd()  # or set to your desired root path

for dir_name in os.listdir(root_dir):
    dir_path = os.path.join(root_dir, dir_name)
    if os.path.isdir(dir_path):
        # Check for "wgan" and "samples" in directory name
        if "wgan" in dir_name.lower() and "samples" in dir_name.lower():
            print(f"Running A* Solvability evaluation on WGAN directory: {dir_path}")
            best_model_json = os.path.join(dir_path, "best_model_info.json")
            if os.path.isfile(best_model_json):
                with open(best_model_json, "r") as f:
                    levels = json.load(f)
                # Run astar_metrics and save output in the same directory
                astar_metrics(
                    levels=levels,
                    num_runs=1,
                    output_json_path=best_model_json,
                    save_name="astar_result.json"
                )
                print(f"A* metrics saved to {os.path.join(dir_path, 'astar_result.json')}")
            else:
                print(f"best_model_info.json not found in {dir_path}, skipping A* metrics.")