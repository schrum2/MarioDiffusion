import os
import re
import json
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from verify_data_complete import find_numbered_directories

# Which modes are valid for which model types
VALID_MODES_BY_TYPE = {
    "conditional": {"real", "random", "short", "long", "real_full"},
    "unconditional": {"short", "long"},
    "wgan": {"short"},
    "fdm": {"real", "random"},
    "MarioGPT": {"short"},
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
    elif "MarioGPT" in model_name:
        return "MarioGPT"
    return "unknown"


def extract_prefix(name):
    if "-unconditional" in name:
        # return re.sub(r"-unconditional\d+", "-unconditional", name)
        return "Mar1and2-unconditional"
    elif "-wgan" in name:
        # return re.sub(r"-wgan\d+", "-wgan", name)
        return "Mar1and2-wgan"
    elif "MarioGPT" in name:
        return "MarioGPT_metrics"
    return name.rstrip("0123456789").rstrip("-_")

# TODO: Add a commandline flag that when set, will indicate that we want to compute metrics with all 7687 real samples. That should reflect here
# Instead of returning evaluation_metrics.json, return evaluation_metrics_full.json
def get_metrics_path(base_dir, mode, plot_file, full_metrics=False):
    model_name = os.path.basename(base_dir)

    if "-conditional-" in model_name: # TODO: handle full case
        if mode in {"short", "long"}:
            # e.g. Mar1and2-conditional-absence5-conditional-samples-short
            cond_dir = f"{base_dir}-unconditional-samples-{mode}"
            return os.path.join(cond_dir, plot_file)
        elif mode in {"real", "random"}:
            # e.g. Mar1and2-conditional-absence5/samples-from-real-Mar1and2-captions/evaluation_metrics.json
            subdir = f"samples-from-{mode}-Mar1and2-captions"
            return os.path.join(base_dir, subdir, plot_file)
        
        elif mode in {"real_full"}:
            if full_metrics:
                subdir = f"samples-from-real-Mar1and2-captions"
                return os.path.join(base_dir,subdir, "evaluation_metrics_full.json")
            else: 
                return None
 

    elif "-fdm-" in model_name: # TODO: handle full case
        # fdm case is always subdir
        if mode in {"real", "random"}:
            subdir = f"samples-from-{mode}-Mar1and2-captions"
            return os.path.join(base_dir, subdir, plot_file)
        elif mode in {"real_full"}:
            if full_metrics:
                subdir = f"samples-from-real-Mar1and2-captions"
                return os.path.join(base_dir,subdir, "evaluation_metrics_full.json")
            else: 
                None

    elif "-unconditional-samples" in model_name:
        # e.g. Mar1and2-unconditional29-unconditional-samples-short
        return os.path.join(base_dir, plot_file)

    elif "-wgan" in model_name:
        return os.path.join(base_dir, plot_file)

    elif "MarioGPT" in model_name:
        return os.path.join(base_dir, f"{mode}_levels", plot_file)
    else:
        print(f"[WARNING] Unknown model type for: {model_name}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Compare models across modes.")
    parser.add_argument("--modes", nargs="+", default=["real", "random"],
                        help="List of modes to compare (e.g., real random short)")
    parser.add_argument("--metric", type=str, default="average_min_edit_distance",
                        help="Metric key in evaluation_metrics.json to plot")
    parser.add_argument("--plot_file", type=str, default="evaluation_metrics.json", help="File with metrics to plot")
    parser.add_argument("--save", action="store_true", help="Stores resulting pdfs in a folder named comparison_plots")
    parser.add_argument("--plot_label", type=str, default=None, help="Label for the outputted plot")
    parser.add_argument("--full_metrics", action="store_true", help="Flag that indicates we will be plotting real_full")
    return parser.parse_args()

def get_bar_color(model_name, mode, mode_list=None, colors=None):
    if "MarioGPT" in model_name:
        return 'red'
    return MODE_COLORS.get(mode, "#cccccc")

# Desired plotting order
MODE_ORDER = ["real_full", "real", "random", "short"]

# Add mode name mapping for legend labels
MODE_DISPLAY_NAMES = {
    "short": "unconditional",
    "real": "real",
    "random": "random",
    "long": "long",
    "real_full": "real (full)",
}

MODE_COLORS = {
    "real_full": "#e78ac3",   # pink
    "real": "#fc8d62",        # orange
    "random": "#8da0cb",      # blue
    "short": "#66c2a5",       # greenish
}

def main():
    args = parse_args()
    
    metric_key = args.metric

    # Ensure modes are in the desired order and present in the input
    modes = [m for m in MODE_ORDER if m in args.modes or (m == "real_full" and args.full_metrics)]
    modes = list(reversed(modes))  # Reverse to control legend/bar order
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
            if mode == "real_full" and model_type not in {"conditional", "fdm"}:
                continue
            if mode not in valid_modes and mode != "real_full":
                continue
            
            for d in dirs:
                metrics_path = get_metrics_path(d, mode, args.plot_file, args.full_metrics)
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
                    print(f"Adding a value to prefix {prefix}")
                else:
                    print(f"[SKIP] {metric_key} missing in: {metrics_path}")

    model_names = list(data.keys())


    from util.naming_conventions import model_name_map as model_list, get_model_name_map_and_order

    model_label_map, clean_labels_sorted = get_model_name_map_and_order()
    
    sorted_models = list(map(lambda x : x[0], model_list))
    sorted_models = list(reversed(sorted_models))
    clean_labels_sorted = list(reversed(clean_labels_sorted))
    
    # Plotting
    bar_width = 0.35
    num_models = len(sorted_models)
    num_modes = len(modes)
    x = [i * (bar_width * num_modes + 0.3) for i in range(num_models)]
    offsets = [(i - (num_modes - 1) / 2) * bar_width for i in range(num_modes)]

    plt.rcParams.update({
        'font.size': 22,
        'axes.labelsize': 22,
        'axes.titlesize': 22,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 16,
        'legend.title_fontsize': 22,
        'figure.titlesize': 22
    })
    
    # ✅ Embed TrueType fonts in the PDF
    plt.rcParams['pdf.fonttype'] = 42

    plt.figure(figsize=(12, 12))

    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']  # Colorblind-friendly, light colors
    has_added_mode_to_legend = {mode: False for mode in modes}  # Track which modes are in legend

    for i, mode in enumerate(modes):
        means = []
        for model in sorted_models:
            values = data[model][mode]
            mean_val = sum(values) / len(values) if values else 0
            means.append(mean_val)

        bar_positions = [xi + offsets[i] for xi in x]

        # Plot bars for each model
        for j, model in enumerate(sorted_models):
            print(f"Processing model {model} in mode {mode}")  # Debug print
            color = get_bar_color(model, mode, modes, colors)
            is_mariogpt = "MarioGPT" in model
            
            # Add mode to legend only once per mode, and never for MarioGPT
            should_add_to_legend = not has_added_mode_to_legend[mode] and not is_mariogpt
            if should_add_to_legend:
                has_added_mode_to_legend[mode] = True
            
            plt.barh(
                bar_positions[j],
                means[j],
                height=bar_width,
                color=color,
                edgecolor='black',
                label=MODE_DISPLAY_NAMES[mode] if should_add_to_legend else None,
                alpha=0.6
            )

            # Scatter plot for individual values
            if args.plot_file == "evaluation_metrics.json":
                values = data[model][mode]
                if values:  # Only plot if we have values
                    y_position = x[j] + offsets[i]
                    plt.scatter(
                        values,
                        [y_position] * len(values),
                        color='black',
                        marker='x',
                        zorder=10
                    )

    plt.yticks(ticks=x, labels=clean_labels_sorted)
    
    if args.plot_label:
        plt.xlabel(args.plot_label, labelpad=10)
    else:
        plt.xlabel(metric_key.replace("_", " ").capitalize(), labelpad=10)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles[::-1],
        labels[::-1],
        #title="Mode",
        loc='best',
        frameon=True,
        edgecolor='black',
    )

    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout(pad=2)

    if args.save:
        renamed_modes = [
            "unconditional" if m == "short"
            else "full real samples" if m == "real_full"
            else m
            for m in modes
        ]
                
        filename = f"comparison_{'_'.join(reversed(renamed_modes))}_{metric_key}.pdf"
        save_path = os.path.join(save_dir, filename)
        
        # Delete existing file if it exists
        if os.path.exists(save_path):
            os.remove(save_path)
            
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0)
        print(f"Plot saved as: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
