import os
import json
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate solvability of Mario levels using A* metrics.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model output directory containing all_levels.json or its subdirectories.")
    parser.add_argument("--save_name", type=str, default="astar_metrics.jsonl", help="Name of the file to save astar metrics results.")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs for astar metrics evaluation.")
    parser.add_argument("--simulator_kwargs", type=str, default="{}", help="JSON string of additional keyword arguments for the simulator.")

    #parser.add_argument("--random_test", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    num_runs = args.num_runs

    # Parse simulator_kwargs from JSON string to dict ONCE
    try:
        simulator_kwargs = json.loads(args.simulator_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to parse simulator_kwargs: {e}. Please provide a valid JSON string.")

    # Parse model name for conditional/unconditional and data
    model_parts = os.path.basename(model_path).split('-')
    data = model_parts[0] if len(model_parts) > 0 else ""
    model_type = model_parts[1] if len(model_parts) > 1 else ""
    is_conditional = (model_type == "conditional")

    # Prepare list of JSONs to evaluate
    json_jobs = []

    # Always add the two unconditional JSONs in the root directory
    for suffix in ["long", "short"]:
        unconditional_dir = f"{model_path}-unconditional-samples-{suffix}"
        unconditional_json = os.path.join(unconditional_dir, "all_levels.json")
        json_jobs.append({"path": unconditional_json, "label": f"unconditional-{suffix}"})

    # If conditional, add the two conditional JSONs in the model directory
    if is_conditional:
        for cond_type in ["random", "real"]:
            cond_dir = os.path.join(model_path, f"samples-from-{cond_type}-{data}-captions")
            cond_json = os.path.join(cond_dir, "all_levels.json")
            json_jobs.append({"path": cond_json, "label": f"conditional-{cond_type}"})

    # Process each JSON
    for job in json_jobs:
        json_path = job["path"]
        label = job["label"]
        if not os.path.exists(json_path):
            raise RuntimeError(f"JSON file not found: {label} JSON not found at {json_path}. Please check the path and try again.")

        scene_caption_data = load_scene_caption_data(json_path)
        if args.save_name:
            # If save_name is provided, use it; otherwise, default to astar_metrics.jsonl
            save_name = args.save_name
        else:
            # Default save name based on label
            save_name = f"astar_result.jsonl"
        # Prepare arguments for astar_metrics, only passing those explicitly set
        astar_args = {
            "levels": scene_caption_data,
            "num_runs": num_runs,
        }
        # Only pass simulator_kwargs if user provided a non-empty string (not just default "{}")
        if args.simulator_kwargs and args.simulator_kwargs.strip() != "{}":
            astar_args["simulator_kwargs"] = simulator_kwargs
        # Only pass save_name if user provided it (not default)
        if args.save_name and args.save_name != "astar_metrics.jsonl":
            astar_args["save_name"] = args.save_name
        # Always pass input_json_path for correct output location
        astar_args["input_json_path"] = json_path

        try:
            results, overall_averages = astar_metrics(**astar_args)
            print(f"Results from astar_metrics for {astar_args.get('save_name', 'astar_result.jsonl')}: {len(results)} levels processed.")
            print(f"Overall averages saved to {os.path.splitext(astar_args.get('save_name', 'astar_result'))[0]}_overall_averages.json")
        except Exception as e:
            raise RuntimeError(f"Error processing {json_path} with label {label}: {e}")