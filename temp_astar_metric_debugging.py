import os
import json
from util.metrics import astar_metrics

def load_scenes_from_json(json_path):
    """
    Loads all scenes from a JSON file where each entry is a dict with a 'scene' key.
    Returns a list of scenes (each a list of lists of ints).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    scenes = [entry['scene'] for entry in data if 'scene' in entry and entry['scene']]
    return scenes

if __name__ == "__main__":
    # Path to your JSON file (update as needed)
    json_path = os.path.join("datasets", "SMB1_LevelsAndCaptions-regular-validate.json")

    print(f"Loading scenes from: {json_path}")
    scenes = load_scenes_from_json(json_path)
    print(f"Loaded {len(scenes)} scenes.")

    # Run astar_metrics for debugging
    try:
        num_runs = 6
        results = astar_metrics(scenes, num_runs=num_runs)
        print(f"Running {num_runs} runs of astar_metrics on loaded scenes.")
        print(f"Results from astar_metrics: {len(results)} scenes processed.\n")
        for idx, metrics in enumerate(results):
            print(f"\nScene {idx + 1}:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
    except Exception as e:
        print("Error running astar_metrics:", e)