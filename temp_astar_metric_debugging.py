import os
import json
from util.metrics import astar_metrics

def load_scene_caption_data(json_path):
    """
    Loads all levels from a JSON file where each entry is a dict with at least a 'scene' key.
    Returns a list of dicts (each with 'scene' and optionally 'caption').
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Only keep levels with a scene
    return [entry for entry in data if 'scene' in entry and entry['scene']]

if __name__ == "__main__":
    # Path to your JSON file
    json_path = r"C:\Users\williamsr\Documents\GitHub\MarioDiffusion\datasets\Mar1and2_LevelsAndCaptions-regular.json"
    save_name = "astar_metrics_results.jsonl"

    print(f"Loading scene-caption data from: {json_path}")
    scene_caption_data = load_scene_caption_data(json_path)
    print(f"Loaded {len(scene_caption_data)} levels.")

    # Run astar_metrics and save results to root directory
    try:
        num_runs = 1
        results = astar_metrics(
            scene_caption_data,
            num_runs=num_runs,
            save_name=save_name
        )
        print(f"Running {num_runs} runs of astar_metrics on loaded levels.")
        print(f"Results from astar_metrics: {len(results[0])} levels processed.\n")
        # for idx, metrics in enumerate(results):
        #     print(f"\nScene {idx + 1}:\n")
        #     for k, v in metrics.items():
        #         print(f"  {k}: {v}\n")
    except Exception as e:
        print("Error running astar_metrics:", e)