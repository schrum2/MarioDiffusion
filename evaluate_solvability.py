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
    parser.add_argument("--save_name", type=str, default="astar_result.jsonl", help="Name of the file to save astar metrics results.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs for astar metrics evaluation.")
    parser.add_argument("--simulator_kwargs", type=str, default="{}", help="JSON string of additional keyword arguments for the simulator.")

    #parser.add_argument("--random_test", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model path does not exist: {model_path}. Please provide a valid path.")

    if args.simulator_kwargs is not None:
        try:
            simulator_kwargs = json.loads(args.simulator_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to parse simulator_kwargs: {e}. Please provide a valid JSON string.")

    # Parse model path name for conditional/unconditional and data
    model_parts = os.path.basename(model_path).split('-')
    data = model_parts[0] if len(model_parts) > 0 else ""
    model_type = model_parts[1] if len(model_parts) > 1 else ""
    is_fdm = (model_type == "fdm")
    is_conditional = (model_type == "conditional") or (model_type == "fdm")
    
    # Prepare list of JSONs to evaluate
    json_jobs = []

    # Always add the two unconditional JSONs in the root directory
    if not is_fdm and "wgan" not in model_path:
        for suffix in ["long", "short"]:
            unconditional_dir = f"{model_path}-unconditional-samples-{suffix}"
            unconditional_json = os.path.join(unconditional_dir, "all_levels.json")
            json_jobs.append({"path": unconditional_json, "label": f"unconditional-{suffix}"})

    # If conditional, add the two conditional JSONs in the model directory
    if is_conditional:
        for real_or_rand in ["random", "real"]:
            cond_dir = os.path.join(model_path, f"samples-from-{real_or_rand}-{data}-captions")
            cond_json = os.path.join(cond_dir, "all_levels.json")
            json_jobs.append({"path": cond_json, "label": f"conditional-{real_or_rand}"})

    # Add WGAN support
    if "wgan" in model_path:
        wgan_samples_dir = model_path + "-samples"
        wgan_json = os.path.join(wgan_samples_dir, "all_levels.json")
        json_jobs.append({"path": wgan_json, "label": "wgan-samples"})

    # Process each JSON
    for job in json_jobs:
        json_path = job["path"]
        label = job["label"]
        if not os.path.exists(json_path):
            raise RuntimeError(f"JSON file not found: {label} JSON not found at {json_path}. Please check the path and try again.")

        scene_caption_data = load_scene_caption_data(json_path)

        sample_limit = 100 # TODO: Make 100 a parameter: how much to limit the output
        if len(scene_caption_data) > sample_limit: 
            increment = len(scene_caption_data) // (sample_limit + 1)
            scene_caption_data = [scene_caption_data[(i+1)*increment] for i in range(sample_limit)]
            if len(scene_caption_data) != sample_limit:
                raise RuntimeError(f"Sample limit mismatch: Expected {sample_limit} samples, got {len(scene_caption_data)} after sampling.")

        # Backup existing astar_result files if they exist
        result_jsonl = os.path.join(os.path.dirname(json_path), args.save_name)
        result_overall = os.path.splitext(result_jsonl)[0] + "_overall_averages.json"
        backup_jsonl = os.path.join(os.path.dirname(json_path), "astar_result.bak.jsonl")
        backup_overall = os.path.join(os.path.dirname(json_path), "astar_result_overall_averages.bak.json")

        if os.path.exists(result_jsonl):
            os.replace(result_jsonl, backup_jsonl)
            print(f"Backed up {result_jsonl} to {backup_jsonl}")
        if os.path.exists(result_overall):
            os.replace(result_overall, backup_overall)
            print(f"Backed up {result_overall} to {backup_overall}")

        try:
            results, overall_averages = astar_metrics(
                levels = scene_caption_data,
                num_runs = args.num_runs,
                simulator_kwargs = simulator_kwargs,
                output_json_path = json_path,
                save_name = args.save_name,
            )
            print(f"Results from astar_metrics for {args.save_name}: {len(results)} levels processed.")
            print(f"Overall averages saved to {os.path.splitext(args.save_name)[0]}_overall_averages.json")
        except Exception as e:
            raise RuntimeError(f"Error processing {json_path} with label {label}: {e}")