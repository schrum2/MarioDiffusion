import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import argparse

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize model statistics with customizable plots.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output", type=str, default="plot.pdf", help="Output plot file name (PDF recommended).")
    parser.add_argument("--plot_type", type=str, choices=["box", "violin", "bar", "scatter"], default="box", help="Type of plot to generate.")
    parser.add_argument("--group_key", type=str, default="group", help="Key to use for grouping models (default: 'group').")
    parser.add_argument("--x_axis", type=str, required=True, help="Used for both naming the key and labeling the x-axis.")
    parser.add_argument("--y_axis", type=str, required=True, help="Used for both naming the key and labeling the y-axis.")
    parser.add_argument("--x_axis_label", type=str, default="", help="Label for the x-axis (default: None).")
    parser.add_argument("--y_axis_label", type=str, default="", help="Label for the y-axis (default: None).")
    parser.add_argument("--font_size", type=int, default=22, help="Base font size for the plot (default: 22).")
    parser.add_argument("--labelsize", type=int, default=24, help="Font size for axes labels (default: 24).")
    parser.add_argument("--xtick_labelsize", type=int, default=20, help="Font size for x-tick labels (default: 20).")
    parser.add_argument("--ytick_labelsize", type=int, default=20, help="Font size for y-tick labels (default: 20).")
    parser.add_argument("--legend_fontsize", type=int, default=20, help="Font size for legend (default: 20).")
    parser.add_argument("--figsize", type=int, nargs=2, default=(10, 10), help="Figure size as width and height in inches (default: 10x10).")

    return parser.parse_args()

def load_data(json_path, group_key):
    records = []
    with open(json_path, "r") as f:
        for line in f:
            group_stats = json.loads(line)
            group = group_stats[group_key]
            for model in group_stats["models"]:
                model[group_key] = group
                records.append(model)
    return pd.DataFrame(records)

def rename_and_order_groups(df, group_key):
    group_name_map = {
        "Mar1and2-conditional-regular": "MLM-regular",
        "Mar1and2-conditional-absence": "MLM-absence",
        "Mar1and2-conditional-negative": "MLM-negative",
        "Mar1and2-conditional-MiniLM-regular": "MiniLM-single-regular",
        "Mar1and2-conditional-MiniLM-absence": "MiniLM-single-absence",
        "Mar1and2-conditional-MiniLM-negative": "MiniLM-single-negative",
        "Mar1and2-conditional-MiniLMsplit-regular": "MiniLM-multiple-regular",
        "Mar1and2-conditional-MiniLMsplit-absence": "MiniLM-multiple-absence",
        "Mar1and2-conditional-MiniLMsplit-negative": "MiniLM-multiple-negative",
        "Mar1and2-conditional-GTE-regular": "GTE-single-regular",
        "Mar1and2-conditional-GTE-absence": "GTE-single-absence",
        "Mar1and2-conditional-GTE-negative": "GTE-single-negative",
        "Mar1and2-conditional-GTEsplit-regular": "GTE-multiple-regular",
        "Mar1and2-conditional-GTEsplit-absence": "GTE-multiple-absence",
        "Mar1and2-conditional-GTEsplit-negative": "GTE-multiple-negative",
        "Mar1and2-unconditional": "Unconditional"
    }
    desired_order = [
        "MLM-regular",
        "MLM-absence",
        "MLM-negative",
        "MiniLM-single-regular",
        "MiniLM-single-absence",
        "MiniLM-single-negative",
        "MiniLM-multiple-regular",
        "MiniLM-multiple-absence",
        "MiniLM-multiple-negative",
        "GTE-single-regular",
        "GTE-single-absence",
        "GTE-single-negative",
        "GTE-multiple-regular",
        "GTE-multiple-absence",
        "GTE-multiple-negative",
        "Unconditional"
    ]
    df[group_key] = df[group_key].replace(group_name_map)
    df[group_key] = pd.Categorical(df[group_key], categories=desired_order, ordered=True)
    df = df.sort_values(group_key)
    return df, desired_order

def main():
    args = parse_args()
    df = load_data(args.input, args.group_key)
    df, desired_order = rename_and_order_groups(df, args.group_key)

    # Update matplotlib settings for better readability
    plt.rcParams.update({
        "font.size": args.font_size,         # Controls default text size
        "axes.labelsize": args.labelsize,    # Axes label font size
        "xtick.labelsize": args.xtick_labelsize,   # X tick label font size
        "ytick.labelsize": args.ytick_labelsize,   # Y tick label font size
        "legend.fontsize": args.legend_fontsize,   # Legend font size
    })

    plt.figure(figsize=args.figsize)

    # Only include groups with data for the selected y_axis
    groups_with_data = [g for g in desired_order if not df[df[args.group_key] == g][args.y_axis].dropna().empty]

    # BOX PLOT
    if args.plot_type == "box":
        df[args.y_axis] = pd.to_numeric(df[args.y_axis], errors="coerce")
        data = [df[df[args.group_key] == g][args.y_axis].dropna() for g in groups_with_data]
        plt.boxplot(data)
        plt.xticks(ticks=range(1, len(groups_with_data)+1), labels=groups_with_data, rotation=45, ha='right')
        plt.xlabel(args.x_axis_label)
        plt.ylabel(args.y_axis_label)
    # VIOLIN PLOT
    elif args.plot_type == "violin":
        df[args.y_axis] = pd.to_numeric(df[args.y_axis], errors="coerce")
        data = [df[df[args.group_key] == g][args.y_axis].dropna() for g in groups_with_data]
        parts = plt.violinplot(data, showmeans=True, showmedians=True)
        plt.xticks(range(1, len(groups_with_data)+1), groups_with_data, rotation=45, ha='right')
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
        y = range(len(groups_with_data))
        plt.barh(y, grouped.reindex(groups_with_data), height=0.4, color="skyblue")
        plt.yticks(y, groups_with_data)
        plt.xlabel(args.x_axis_label)
        plt.ylabel(args.y_axis_label)
        for i, group in enumerate(groups_with_data):
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
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
