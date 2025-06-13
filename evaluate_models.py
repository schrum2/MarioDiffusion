import os
import re
import json
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from verify_data_complete import find_numbered_directories

# Which modes are valid for which model types
VALID_MODES_BY_TYPE = {
    "conditional": {"real", "random", "short", "long"},
    "unconditional": {"short", "long"},
    "wgan": {"short"},
    "fdm": {"real", "random"},
}

def detect_model_type(model_name):
    if "-conditional-" in model_name:
        return "conditional"
    elif "-unconditional" in model_name:
        return "unconditional"
    elif "-wgan" in model_name:
        return "wgan"
    elif "-fdm-" in model_name:
        return "fdm"
    return "unknown"


def strip_common_prefix(strings):
    if not strings:
        return strings
    common_prefix = os.path.commonprefix(strings)
    return [s[len(common_prefix):] for s in strings], common_prefix

def extract_prefix(name):
    if "-unconditional" in name:
        return re.sub(r"-unconditional\d+", "-unconditional", name)
    elif "-wgan" in name:
        return re.sub(r"-wgan\d+", "-wgan", name)
    return name.rstrip("0123456789").rstrip("-_")

def get_metrics_path(base_dir, mode):
    model_name = os.path.basename(base_dir)

    if "-conditional-" in model_name:
        if mode in {"short", "long"}:
            # e.g. Mar1and2-conditional-absence5-conditional-samples-short
            cond_dir = f"{base_dir}-conditional-samples-{mode}"
            return os.path.join(cond_dir, "evaluation_metrics.json")
        else:
            # e.g. Mar1and2-conditional-absence5/samples-from-real-Mar1and2-captions/evaluation_metrics.json
            subdir = f"samples-from-{mode}-Mar1and2-captions"
            return os.path.join(base_dir, subdir, "evaluation_metrics.json")

    elif "-fdm-" in model_name:
        # fdm case is always subdir
        subdir = f"samples-from-{mode}-Mar1and2-captions"
        return os.path.join(base_dir, subdir, "evaluation_metrics.json")

    elif "-unconditional" in model_name:
        # e.g. Mar1and2-unconditional29-unconditional-samples-short
        return os.path.join(base_dir, "evaluation_metrics.json")

    elif "-wgan" in model_name:
        return os.path.join(base_dir, "evaluation_metrics.json")

    else:
        print(f"[WARNING] Unknown model type for: {model_name}")
        return None

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

    parent_dir = os.path.dirname(numbered_dirs[0][0])
    save_dir = os.path.join(parent_dir, "comparison_plots")
    os.makedirs(save_dir, exist_ok=True)

    grouped = defaultdict(list)
    for dir_path, num, dir_type in numbered_dirs:
        prefix = extract_prefix(dir_path)
        grouped[prefix].append(dir_path)

    data = defaultdict(lambda: defaultdict(list))

    for prefix, dirs in grouped.items():
        model_type = detect_model_type(prefix)
        valid_modes = VALID_MODES_BY_TYPE.get(model_type, set())
    
        for mode in modes:
            if mode not in valid_modes:
                continue
            
            for d in dirs:
                # Modify to correct model path for fdm (and later for wgan)

                metrics_path = get_metrics_path(d, mode)
                if not metrics_path or not os.path.exists(metrics_path):
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

    # Step 1: Strip common prefix
    clean_labels, removed_prefix = strip_common_prefix(model_names)
    print(f"Removed common prefix: '{removed_prefix}'")

    # Step 2: Renaming logic, REPLACE WITH UTIL SCRIPT THAT HAS NAMING CONVENTIONS
    def rename_model_label(label):
        if label in {"regular", "absence", "negative"}:
            return f"MLM-{label}"
        elif "split" in label:
            return label.replace("split", "multiple")
        else:
            parts = label.split("-")
            if len(parts) == 2:
                return f"{parts[0]}-single-{parts[1]}"
            else:
                return f"{label}-single"

    model_label_map = {original: rename_model_label(cleaned) for original, cleaned in zip(model_names, clean_labels)}
    sorted_models = sorted(model_names)
    clean_labels_sorted = [model_label_map[m] for m in sorted_models]

    # Plotting
    bar_width = 0.35
    num_models = len(sorted_models)
    num_modes = len(modes)
    x = [i * (bar_width * num_modes + 0.3) for i in range(num_models)]
    offsets = [(i - (num_modes - 1) / 2) * bar_width for i in range(num_modes)]

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.title_fontsize': 14,
        'figure.titlesize': 18
    })
    
    # ✅ Embed TrueType fonts in the PDF
    plt.rcParams['pdf.fonttype'] = 42

    plt.figure(figsize=(8, 8))

    colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # Colorblind-friendly, light colors

    for i, mode in enumerate(modes):
        means = []
        for model in sorted_models:
            values = data[model][mode]
            mean_val = sum(values) / len(values) if values else 0
            means.append(mean_val)

        bar_positions = [xi + offsets[i] for xi in x]
        plt.barh(
            bar_positions,
            means,
            height=bar_width,
            color=colors[i % len(colors)],
            edgecolor='black',
            label=mode,
            alpha=0.6
        )

        for j, model in enumerate(sorted_models):
            values = data[model][mode]
            y_positions = [x[j] + offsets[i]] * len(values)
            plt.scatter(
                values,
                y_positions,
                color='black',
                marker='x',
                zorder=10
            )

    plt.yticks(ticks=x, labels=clean_labels_sorted)
    plt.xlabel(metric_key.replace("_", " ").capitalize(), labelpad=10)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles[::-1],
        labels[::-1],
        title="Mode",
        loc='best',
        frameon=True,
        edgecolor='black'
    )

    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout(pad=2)

    if args.save:
        filename = f"comparison_{'_'.join(reversed(modes))}_{metric_key}.pdf"
        save_path = os.path.join(save_dir, filename)
        
        # Delete existing file if it exists
        if os.path.exists(save_path):
            os.remove(save_path)
            
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved as: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
