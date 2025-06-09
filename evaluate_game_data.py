import json
import argparse
import os
from pathlib import Path
import glob
from util.metrics import average_min_edit_distance

def parse_args():
    
    parser = argparse.ArgumentParser(description="Evaluate data used to train and test a model")
    parser.add_argument("--game", required=True, default=None, help="Game data to be evaluated")
    return parser.parse_args()
    
def main():
    args = parse_args()
    game_prefix = args.game
    output_file = Path(f"{game_prefix}_evaluation_game_data_results.json")
    
    # Path to the datasets directory (relative to your current dir)
    datasets_dir = os.path.join("datasets")
    
    # Build the search pattern for json files with the given prefix
    pattern = os.path.join(datasets_dir, f"{game_prefix}*.json")
    
    # Get all matching files
    matching_files = glob.glob(pattern)
    
    # Filter out files that contain "RandomTest"
    filtered_files = [
        f for f in matching_files
        if "RandomTest" not in os.path.basename(f)
        and os.path.basename(f) != f"{game_prefix}_Levels.json"
        and "tiles" not in os.path.basename(f)
        and "ValidationCaptions" not in os.path.basename(f)
    ]
    
    results = {}
    
    # Process each dataset file
    for file in filtered_files:
        print(f"Processing: {os.path.basename(file)}")
        with open(file, "r") as f:
            data = json.load(f)
            
        # Calculate the average minimum edit distance
        
        # Extract scences to get the levels
        levels = [entry["scene"] for entry in data if "scene" in entry]
        
        # Compute average min edit distance on current file
        avg_edit_dist = average_min_edit_distance(levels)
        
        results[os.path.basename(file)] = avg_edit_dist
        print(f"Average Min Edit Distance: {avg_edit_dist:.4f}")
        
    # Save results to a JSON file once all files are evaluated
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
    
if __name__ == "__main__":
    main()
        