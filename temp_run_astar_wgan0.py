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
    # Paths to your JSON files
    json_path1 = r"G:\.shortcut-targets-by-id\1C1aB0CgC2ozojftq5eagcmSg65fS7ttg\SURF Artifacts\EXPERIMENTS\MarioDiffusion-FIX-SMB-DATA\Mar1and2-wgan0-samples\all_levels.json"
    
    for json_path in [(json_path1)]:
        print(f"\nLoading scene-caption data from: {json_path}")
        scene_caption_data = load_scene_caption_data(json_path)
        print(f"Loaded {len(scene_caption_data)} levels.")
        num_runs = 1

        try:
            results, overall_averages = astar_metrics(
                scene_caption_data,
                num_runs=num_runs
            )
            print(f"Results for {json_path}:")
        except Exception as e:
            print(f"Error running astar_metrics for {json_path}:", e)