import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import argparse
from util.naming_conventions import get_model_name_map_and_order

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize model statistics with customizable plots.\n")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file.\n")
    parser.add_argument("--output", type=str, default="plot.pdf", help="Output plot file name (PDF recommended).\n")
    parser.add_argument("--plot_type", type=str, choices=["box", "violin", "bar", "scatter", "horizontal_box"], default="horizontal_box", help="Type of plot to generate.\n")
    parser.add_argument("--group_key", type=str, default="group", help="Key to use for grouping models (default: 'group').\n")
    parser.add_argument("--x_axis", type=str, required=True, help="Used for both naming the key and labeling the x-axis.\n")
    parser.add_argument("--y_axis", type=str, required=True, help="Used for both naming the key and labeling the y-axis.\n")
    parser.add_argument("--x_axis_label", type=str, default="", help="Label for the x-axis (default: None).\n")
    parser.add_argument("--y_axis_label", type=str, default="", help="Label for the y-axis (default: None).\n")
    parser.add_argument("--font_size", type=int, default=22, help="Base font size for the plot (default: 22).\n")
    parser.add_argument("--labelsize", type=int, default=24, help="Font size for axes labels (default: 24).\n")
    parser.add_argument("--xtick_labelsize", type=int, default=20, help="Font size for x-tick labels (default: 20).\n")
    parser.add_argument("--ytick_labelsize", type=int, default=20, help="Font size for y-tick labels (default: 20).\n")
    parser.add_argument("--legend_fontsize", type=int, default=20, help="Font size for legend (default: 20).\n")
    parser.add_argument("--figsize", type=int, nargs=2, default=(10, 10), help="Figure size as width and height in inches (default: 10x10).\n")
    parser.add_argument("--x_tick_rotation", type=int, default=0, help="Rotation angle for axis labels (default: 0).\n")
    parser.add_argument("--x_markers_on_bar_plot", action='store_true', help="If set, scatter points will be plotted on top of the bar plot.\n")
    parser.add_argument("--x_marker_data_on_bar_plot", type=str, default=None, help="Choose data other than that provided for the required --x_axis arg.\n")
    parser.add_argument("--stacked_bar_for_mlm", action='store_true', help="If set, MLM groups with mlm_mean/cond_mean will be shown as stacked bars.\n")
    parser.add_argument("--convert_time_to_hours", action='store_true', help="If set, time values will be converted to hours.\n")
    parser.add_argument("--font", type=str, default="Courier New", help="Font family for the plo e.g. \"Times New Roman\", \"Palatino\", \"Garamond\" (default: 'Courier New').\n")
    return parser.parse_args()

# GPT-4.1 suggested a more robust JSON loading function that can handle both JSON arrays and JSONL files.
# This function attempts to load data from a JSON file, and if it fails, it tries to read it as a JSONL file.
def load_data(json_path, group_key):
    records = []
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            for group_stats in data:
                group = group_stats.get(group_key)
                if "models" in group_stats and isinstance(group_stats["models"], list):
                    for model in group_stats["models"]:
                        model[group_key] = group
                        records.append(model)
                elif "individual_times" in group_stats and isinstance(group_stats["individual_times"], list):
                    for val in group_stats["individual_times"]:
                        records.append({group_key: group, "individual_times": val, **{k: v for k, v in group_stats.items() if k not in [group_key, "individual_times"]}})
                else:
                    # Generic: flatten any list-of-numbers key (except 'models')
                    for k, v in group_stats.items():
                        if k != "models" and isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                            for val in v:
                                records.append({group_key: group, k: val, **{kk: vv for kk, vv in group_stats.items() if kk not in [group_key, k]}})
                            break
                    else:
                        records.append(group_stats)
            return pd.DataFrame(records)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
    except Exception:
        # Fallback: try JSONL
        with open(json_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    group_stats = json.loads(line)
                    group = group_stats.get(group_key)
                    if "models" in group_stats and isinstance(group_stats["models"], list):
                        for model in group_stats["models"]:
                            model[group_key] = group
                            records.append(model)
                    elif "individual_times" in group_stats and isinstance(group_stats["individual_times"], list):
                        for val in group_stats["individual_times"]:
                            records.append({group_key: group, "individual_times": val, **{k: v for k, v in group_stats.items() if k not in [group_key, "individual_times"]}})
                    else:
                        # Generic: flatten any list-of-numbers key (except 'models')
                        for k, v in group_stats.items():
                            if k != "models" and isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                                for val in v:
                                    records.append({group_key: group, k: val, **{kk: vv for kk, vv in group_stats.items() if kk not in [group_key, k]}})
                                break
                        else:
                            records.append(group_stats)
        return pd.DataFrame(records)

def rename_and_order_groups(df, group_key):
    group_name_map, desired_order = get_model_name_map_and_order()
    df[group_key] = df[group_key].replace(group_name_map)
    df[group_key] = pd.Categorical(df[group_key], categories=desired_order, ordered=True)
    df = df.sort_values(group_key)
    return df, desired_order

def main():
    args = parse_args()
    df = load_data(args.input, args.group_key)
    df, desired_order = rename_and_order_groups(df, args.group_key)

    # General time conversion: convert all numeric columns and numeric lists from seconds to hours
    if args.convert_time_to_hours:
        for col in df.columns:
            # Convert numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col] / 3600.0
            # Convert columns with lists of numbers
            elif df[col].apply(lambda x: isinstance(x, list) and all(isinstance(i, (int, float)) for i in x) if pd.notnull(x) else False).any():
                df[col] = df[col].apply(lambda x: [i / 3600.0 for i in x] if isinstance(x, list) and all(isinstance(i, (int, float)) for i in x) else x)

    # Update matplotlib settings for better readability
    plt.rcParams.update({
        "font.size": args.font_size,         # Controls default text size
        "axes.labelsize": args.labelsize,    # Axes label font size
        "xtick.labelsize": args.xtick_labelsize,   # X tick label font size
        "ytick.labelsize": args.ytick_labelsize,   # Y tick label font size
        "legend.fontsize": args.legend_fontsize,   # Legend font size
        "font.family": args.font, # Use a monospaced font for better alignment
    })

    plt.figure(figsize=args.figsize)

    # Only include groups with data for the selected y_axis
    groups_with_data = [g for g in desired_order if not df[df[args.group_key] == g][args.y_axis].dropna().empty]

    # VERTICAL BOX PLOT
    if args.plot_type == "box":
        df[args.y_axis] = pd.to_numeric(df[args.y_axis], errors="coerce")
        data = [df[df[args.group_key] == g][args.y_axis].dropna() for g in groups_with_data]
        plt.boxplot(data)
        plt.xticks(ticks=range(1, len(groups_with_data)+1), labels=groups_with_data, rotation=args.x_tick_rotation, ha='right')
        plt.xlabel(args.x_axis_label)
        plt.ylabel(args.y_axis_label)
    # HORIZONTAL BOX PLOT
    elif args.plot_type == "horizontal_box":
        df[args.x_axis] = pd.to_numeric(df[args.x_axis], errors="coerce")
        data = [df[df[args.group_key] == g][args.x_axis].dropna() for g in groups_with_data]
        groups_reversed = groups_with_data[::-1]
        y = range(len(groups_reversed))
        y = [i + 1 for i in y]  # Adjust y-ticks to start from 1 for horizontal box plot
        plt.boxplot(data, vert=False)
        plt.yticks(y, labels=groups_reversed, rotation=args.x_tick_rotation, ha='right')
        plt.ylabel(args.y_axis_label)
        plt.xlabel(args.x_axis_label)
    # VIOLIN PLOT
    elif args.plot_type == "violin":
        df[args.y_axis] = pd.to_numeric(df[args.y_axis], errors="coerce")
        data = [df[df[args.group_key] == g][args.y_axis].dropna() for g in groups_with_data]
        parts = plt.violinplot(data, showmeans=True, showmedians=True)
        plt.xticks(range(1, len(groups_with_data)+1), groups_with_data, rotation=args.x_tick_rotation, ha='right')
        plt.xlabel(args.x_axis_label)
        plt.ylabel(args.y_axis_label)
        if 'cmedians' in parts:
            parts['cmedians'].set_color('red')
            parts['cmedians'].set_linewidth(1.5)
        if 'cmeans' in parts:
            parts['cmeans'].set_color('blue')
            parts['cmeans'].set_linewidth(3)
        median_line = mlines.Line2D([], [], color='red', linewidth=3, label='Median')
        mean_line = mlines.Line2D([], [], color='blue', linewidth=3, label='Mean')
        plt.legend(handles=[median_line, mean_line], loc='lower left')
    # HORIZANTAL BAR PLOT
    elif args.plot_type == "bar":
        df[args.x_axis] = pd.to_numeric(df[args.x_axis], errors="coerce")
        grouped = df.groupby(args.group_key, observed=False)[args.x_axis].mean()
        groups_reversed = groups_with_data[::-1]
        y = range(len(groups_reversed))
        if args.stacked_bar_for_mlm:
            # Prepare data for stacked bars
            mlm_groups = [g for g in groups_reversed if g in ["MLM-regular", "MLM-absence", "MLM-negative"]]
            other_groups = [g for g in groups_reversed if g not in mlm_groups]
            mlm_means = []
            cond_means = []
            bar_labels = []
            bar_positions = []
            single_means = []
            single_positions = []
            for i, g in enumerate(groups_reversed):
                row = df[df[args.group_key] == g].iloc[0] if not df[df[args.group_key] == g].empty else None
                if g in mlm_groups and row is not None and "mlm_mean" in row and "cond_mean" in row:
                    mlm_means.append(row["mlm_mean"])
                    cond_means.append(row["cond_mean"])
                    bar_labels.append(g)
                    bar_positions.append(i)
                elif row is not None and "mean" in row:
                    single_means.append(row["mean"])
                    single_positions.append(i)
            # Plot stacked bars for MLM groups
            plt.barh(bar_positions, mlm_means, height=0.4, color="red", label="Language Model")
            plt.barh(bar_positions, cond_means, height=0.4, left=mlm_means, color="skyblue", label="Level Model")
            plt.grid(axis='x', which='both', linestyle='--', alpha=0.5)
            # Plot single bars for other groups
            plt.barh(single_positions, single_means, height=0.4, color="skyblue")
            plt.yticks(y, groups_reversed)
            plt.xlabel(args.x_axis_label)
            plt.ylabel(args.y_axis_label)
            plt.xticks(rotation=args.x_tick_rotation)
            plt.legend()
            if args.x_markers_on_bar_plot:
                if args.x_marker_data_on_bar_plot:
                    x_marker_data = df[args.x_marker_data_on_bar_plot].dropna()
                    for i, group in enumerate(groups_reversed):
                        points = df[df[args.group_key] == group][args.x_marker_data_on_bar_plot].dropna()
                        plt.scatter(points, [i]*len(points), color='k', alpha=0.6, s=30, marker='x', label='_nolegend_')
                        
                else:
                    plt.gca().set_xlim(left=0)
                    for i, group in enumerate(groups_reversed):
                        points = df[df[args.group_key] == group][args.x_axis].dropna()
                        plt.scatter(points, [i]*len(points), color='k', alpha=0.6, s=30, marker='x', label='_nolegend_')
        else:
            plt.barh(y, grouped.reindex(groups_reversed), height=0.4, color="skyblue")
            plt.yticks(y, groups_reversed)
            plt.xlabel(args.x_axis_label)
            plt.ylabel(args.y_axis_label)
            plt.xticks(rotation=args.x_tick_rotation)
            if args.x_markers_on_bar_plot:
                if args.x_marker_data_on_bar_plot:
                    x_marker_data = df[args.x_marker_data_on_bar_plot].dropna()
                    for i, group in enumerate(groups_reversed):
                        points = df[df[args.group_key] == group][args.x_marker_data_on_bar_plot].dropna()
                        plt.scatter(points, [i]*len(points), color='k', alpha=0.6, s=30, marker='x', label='_nolegend_')
                else:
                    plt.gca().set_xlim(left=0)
                    for i, group in enumerate(groups_reversed):
                        points = df[df[args.group_key] == group][args.x_axis].dropna()
                        plt.scatter(points, [i]*len(points), color='k', alpha=0.6, s=30, marker='x', label='_nolegend_')
    # SCATTER PLOT
    elif args.plot_type == "scatter":
        color_map = plt.get_cmap('Set2', len(groups_with_data))
        for i, g in enumerate(groups_with_data):
            subset = df[df[args.group_key] == g]
            if args.y_axis in subset.columns:
                plt.scatter(subset[args.x_axis], subset[args.y_axis], label=g, s=80, color=color_map(i))
        plt.xlabel(args.x_axis_label)
        plt.ylabel(args.y_axis_label)
        plt.xticks(rotation=args.x_tick_rotation)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
