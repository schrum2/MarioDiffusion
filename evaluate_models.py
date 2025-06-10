import os
import json
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from verify_data_complete import find_numbered_directories

# Want to go through each directory with a shared prefix (just different # or version of that model)
# Extract and plot the average min edit distance from each of these files. Use a barplot for avg and an x for each of the actual points
# The outer loop should be for each prefix

def extract_prefix(name):
    """Extract the prefix before the final number from a directory name."""
    return name.rstrip("0123456789").rstrip("-_")

def get_metrics_path(base_dir, mode):
    """Return the path to evaluation_metrics.json based on mode."""
    subdir = f"samples-from-{mode}-Mar1and2-captions"
    return os.path.join(base_dir, subdir, "evaluation_metrics.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot avg min edit distance from diffusion models.")
    parser.add_argument("--mode", choices=["real", "random"], required=True,
                        help="Which caption sampling mode to evaluate: 'real' or 'random'")
    return parser.parse_args()

def main():
    args = parse_args()
    mode = args.mode
    print(f"Evaluating '{mode}' caption samples")
    
    numbered_dirs = find_numbered_directories()
    if not numbered_dirs:
        print("No matching directories found.")
        return
    
    # Group directories by shared prefix
    grouped = defaultdict(list)
    for dir_path, num, dir_type in numbered_dirs:
        prefix = extract_prefix(dir_path)
        grouped[prefix].append(dir_path)
        
    all_means = []
    all_labels = []

    plt.figure(figsize=(12, 6))

    for i, (prefix, dirs) in enumerate(sorted(grouped.items())):
        values = []
        for d in dirs:
            metrics_path = get_metrics_path(d, mode)
            if not os.path.exists(metrics_path):
                print(f"[SKIP] Missing evaluation_metrics.json in: {metrics_path}")
                continue
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            except Exception as e:
                print(f"[SKIP] Failed to read JSON from: {metrics_path}. Error: {e}")
                continue

            val = metrics.get("average_min_edit_distance")
            if val is not None:
                values.append(val)
            else:
                print(f"[SKIP] 'average_min_edit_distance' not found in: {metrics_path}")

        if values:
            mean_val = sum(values) / len(values)
            all_means.append(mean_val)
            all_labels.append(prefix)

            # Plot bar for mean
            plt.bar(i, mean_val, color="skyblue", edgecolor="black", linewidth=2, label="Average" if i == 0 else "")
            # Plot x's for each value
            plt.scatter([i]*len(values), values, color="black", marker="x", label="Run" if i == 0 else "")

    plt.xticks(range(len(all_labels)), all_labels, rotation=45, ha='right')
    plt.ylabel("Average Min Edit Distance")
    plt.title("Performance Comparison of Conditional Diffusion Models")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.show()

        

if __name__ == "__main__":
    main()