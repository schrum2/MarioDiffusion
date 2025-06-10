import os
import json
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from verify_data_complete import find_numbered_directories

def strip_common_prefix(strings):
    """Strip the longest common prefix from a list of strings."""
    if not strings:
        return strings
    common_prefix = os.path.commonprefix(strings)
    return [s[len(common_prefix):] for s in strings], common_prefix

def extract_prefix(name):
    """Extract the prefix before the final number from a directory name."""
    return name.rstrip("0123456789").rstrip("-_")

def get_metrics_path(base_dir, mode):
    """Return the path to evaluation_metrics.json based on mode."""
    subdir = f"samples-from-{mode}-Mar1and2-captions"
    return os.path.join(base_dir, subdir, "evaluation_metrics.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Compare models across modes.")
    parser.add_argument("--modes", nargs="+", default=["real", "random"],
                        help="List of modes to compare (e.g., real random short)")
    parser.add_argument("--metric", type=str, default="average_min_edit_distance",
                        help="Metric key in evaluation_metrics.json to plot")
    return parser.parse_args()

def main():
    args = parse_args()
    modes = args.modes
    metric_key = args.metric
    print(f"Comparing modes: {modes}")
    
    numbered_dirs = find_numbered_directories()
    if not numbered_dirs:
        print("No matching directories found.")
        return

    # Group directories by model prefix
    grouped = defaultdict(list)
    for dir_path, num, dir_type in numbered_dirs:
        prefix = extract_prefix(dir_path)
        grouped[prefix].append(dir_path)

    # Collect data: model -> mode -> list of values
    data = defaultdict(lambda: defaultdict(list))

    for prefix, dirs in grouped.items():
        for mode in modes:
            for d in dirs:
                metrics_path = get_metrics_path(d, mode)
                if not os.path.exists(metrics_path):
                    print(f"[SKIP] Missing: {metrics_path}")
                    continue
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                except Exception as e:
                    print(f"[SKIP] Failed to read {metrics_path}: {e}")
                    continue
                val = metrics.get(metric_key)
                if val is not None:
                    data[prefix][mode].append(val)
                else:
                    print(f"[SKIP] {metric_key} missing in: {metrics_path}")

    # Strip common prefix from model labels
    model_names = list(data.keys())
    clean_labels, removed_prefix = strip_common_prefix(model_names)
    print(f"Removed common prefix: '{removed_prefix}'")

    # Sort models alphabetically (or you can sort by real performance etc.)
    sorted_models = sorted(model_names)
    clean_labels_sorted = [label for _, label in sorted(zip(model_names, clean_labels))]

    # Bar setup
    bar_width = 0.35
    num_models = len(sorted_models)
    num_modes = len(modes)
    x = range(num_models)
    offsets = [(i - (num_modes - 1) / 2) * bar_width for i in range(num_modes)]

    plt.figure(figsize=(max(10, num_models * 1.5), 6))

    for i, mode in enumerate(modes):
        means = []
        for model in sorted_models:
            values = data[model][mode]
            mean_val = sum(values) / len(values) if values else 0
            means.append(mean_val)
        bar_positions = [xi + offsets[i] for xi in x]
        plt.bar(bar_positions, means, width=bar_width, label=mode, edgecolor="black")

        # Plot individual "x" marks
        for j, model in enumerate(sorted_models):
            values = data[model][mode]
            x_positions = [x[j] + offsets[i]] * len(values)
            plt.scatter(x_positions, values, color="black", marker="x")

    plt.xticks(ticks=x, labels=clean_labels_sorted, rotation=45, ha='right')
    plt.ylabel("Average Min Edit Distance")
    plt.title("Model Comparison: Real vs Random Captions")
    plt.legend(title="Mode")
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.show()

if __name__ == "__main__":
    main()
