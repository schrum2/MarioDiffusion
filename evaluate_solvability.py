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
    json_path1 = r"C:\Users\haganb\Documents\GitHub\MarioDiffusion\datasets\Mar1and2_LevelsAndCaptions-regular.json"
    json_path2 = r"C:\Users\haganb\Documents\GitHub\MarioDiffusion\datasets\Mar1and2_RandomTest-regular.json"
    save_name1 = "astar_metrics_Mar1and2_LevelsAndCaptions-regular.json"
    save_name2 = "astar_metrics_Mar1and2_RandomTest-regular.json"

    for json_path, save_name in [(json_path1, save_name1), (json_path2, save_name2)]:
        print(f"\nLoading scene-caption data from: {json_path}")
        scene_caption_data = load_scene_caption_data(json_path)
        print(f"Loaded {len(scene_caption_data)} levels.")

        try:
            results, overall_averages = astar_metrics(
                scene_caption_data,
                save_name=save_name
            )
            print(f"Results from astar_metrics for {save_name}: {len(results)} levels processed.")
            print(f"Overall averages saved to {os.path.splitext(save_name)[0]}_overall_averages.json")
        except Exception as e:
            print(f"Error running astar_metrics for {json_path}:", e)