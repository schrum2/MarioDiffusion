import os
import json
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from verify_data_complete import find_numbered_directories

def strip_common_prefix(strings):
    if not strings:
        return strings
    common_prefix = os.path.commonprefix(strings)
    return [s[len(common_prefix):] for s in strings], common_prefix

def extract_prefix(name):
    return name.rstrip("0123456789").rstrip("-_")

def get_metrics_path(base_dir, mode):
    """
    Return the path to evaluation_metrics.json based on mode.
    Supports 'real', 'random', and 'short'.
    """
    if mode == "short" or mode == "long":
        parent_dir = os.path.dirname(base_dir)
        model_name = os.path.basename(base_dir)
        uncond_dir = os.path.join(parent_dir, f"{model_name}-unconditional-samples-{mode}")
        return os.path.join(uncond_dir, "evaluation_metrics.json")
    else:
        subdir = f"samples-from-{mode}-Mar1and2-captions"
        return os.path.join(base_dir, subdir, "evaluation_metrics.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Compare models across modes.")
    parser.add_argument("--modes", nargs="+", default=["real", "random"],
                        help="List of modes to compare (e.g., real random short)")
    parser.add_argument("--metric", type=str, default="average_min_edit_distance",
                        help="Metric key in evaluation_metrics.json to plot")
    parser.add_argument("--save", action="store_true", help="Stores resulting pdfs in a folder named comparison_plots")
    return parser.parse_args()

def main():
    args = parse_args()
    modes = list(reversed(args.modes))  # Reverse to control legend/bar order
    metric_key = args.metric
    print(f"Comparing modes: {modes}")

    numbered_dirs = find_numbered_directories()
    if not numbered_dirs:
        print("No matching directories found.")
        return
    
    # Determine where to save plots (same level as the models)
    parent_dir = os.path.dirname(numbered_dirs[0][0])  # directory of the first model
    save_dir = os.path.join(parent_dir, "comparison_plots") # create the folder where pdf graphs will be
    os.makedirs(save_dir, exist_ok=True)

    grouped = defaultdict(list)
    for dir_path, num, dir_type in numbered_dirs:
        prefix = extract_prefix(dir_path)
        grouped[prefix].append(dir_path)

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

    model_names = list(data.keys())
    clean_labels, removed_prefix = strip_common_prefix(model_names)
    print(f"Removed common prefix: '{removed_prefix}'")

    sorted_models = sorted(model_names)
    clean_labels_sorted = [label for _, label in sorted(zip(model_names, clean_labels))]

    bar_width = 0.35
    num_models = len(sorted_models)
    num_modes = len(modes)
    x = [i * (bar_width * num_modes + 0.3) for i in range(num_models)]  # Add spacing between model groups
    offsets = [(i - (num_modes - 1) / 2) * bar_width for i in range(num_modes)]

    plt.figure(figsize=(max(10, num_models * 1.5), 6))

    for i, mode in enumerate(modes):
        means = []
        for model in sorted_models:
            values = data[model][mode]
            mean_val = sum(values) / len(values) if values else 0
            means.append(mean_val)

        bar_positions = [xi + offsets[i] for xi in x]
        plt.barh(bar_positions, means, height=bar_width, label=mode, edgecolor="black")

        # Add individual "x" markers
        for j, model in enumerate(sorted_models):
            values = data[model][mode]
            y_positions = [x[j] + offsets[i]] * len(values)
            plt.scatter(values, y_positions, color="black", marker="x")

    plt.yticks(ticks=x, labels=clean_labels_sorted)
    plt.xlabel(metric_key.replace("_", " ").capitalize())
    plt.title(f"Model Comparison: {', '.join(reversed(modes))}")  # Show original order in title
    #plt.legend(title="Mode", loc='best')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title="Mode", loc='best')
    plt.tight_layout()
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)

    if args.save:
        filename = f"comparison_{'_'.join(reversed(modes))}_{metric_key}.pdf"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved as: {save_path}")
    
    #plt.show()

if __name__ == "__main__":
    main()
