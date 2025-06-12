import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42  # Embed TrueType fonts in PDF
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.lines as mlines
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize model statistics with customizable plots.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output", type=str, default="plot.pdf", help="Output plot file name (PDF recommended).")
    parser.add_argument("--plot_type", type=str, choices=["box", "violin", "bar", "scatter"], default="box", help="Type of plot to generate.")
    return parser.parse_args()

def load_data(json_path):
    records = []
    with open(json_path, "r") as f:
        for line in f:
            group_stats = json.loads(line)
            group = group_stats["group"]
            for model in group_stats["models"]:
                model["group"] = group
                records.append(model)
    return pd.DataFrame(records)

def rename_and_order_groups(df):
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
    df["group"] = df["group"].replace(group_name_map)
    df["group"] = pd.Categorical(df["group"], categories=desired_order, ordered=True)
    df = df.sort_values("group")
    return df, desired_order

def main():
    args = parse_args()
    df = load_data(args.input)
    df, desired_order = rename_and_order_groups(df)
    df["best_epoch"] = pd.to_numeric(df["best_epoch"], errors="coerce")

    # Update matplotlib settings for better readability
    plt.rcParams.update({
        "font.size": 22,         # Controls default text size
        "axes.titlesize": 24,    # Axes title font size
        "axes.labelsize": 24,    # Axes label font size
        "xtick.labelsize": 20,   # X tick label font size
        "ytick.labelsize": 20,   # Y tick label font size
        "legend.fontsize": 20,   # Legend font size
        "figure.titlesize": 26   # Figure title font size (if used)
    })

    plt.figure(figsize=(10, 10))
    if args.plot_type == "box":
        data = [df[df["group"] == g]["best_epoch"].dropna() for g in desired_order]
        plt.boxplot(data)
        plt.xticks(ticks=range(1, len(desired_order)+1), labels=desired_order, rotation=45, ha='right')
        plt.xlabel("Model Group")
        plt.ylabel("Best Epoch")
    elif args.plot_type == "violin":
        data = [df[df["group"] == g]["best_epoch"].dropna() for g in desired_order]
        parts = plt.violinplot(data, showmeans=True, showmedians=True)
        plt.xticks(range(1, len(desired_order)+1), desired_order, rotation=45, ha='right')
        plt.xlabel("Model Group")
        plt.ylabel("Best Epoch")
        if 'cmedians' in parts:
            parts['cmedians'].set_color('red')
            parts['cmedians'].set_linewidth(1.5)
        if 'cmeans' in parts:
            parts['cmeans'].set_color('blue')
            parts['cmeans'].set_linewidth(3)
        median_line = mlines.Line2D([], [], color='red', linewidth=3, label='Median')
        mean_line = mlines.Line2D([], [], color='blue', linewidth=3, label='Mean')
        plt.legend(handles=[median_line, mean_line], loc='lower left')
    elif args.plot_type == "bar":
        grouped = df.groupby("group")["best_epoch"].mean()
        y = range(len(desired_order))
        plt.barh(y, grouped.reindex(desired_order), height=0.4, color="skyblue")
        plt.yticks(y, desired_order)
        plt.xlabel("Best Epoch")
        plt.ylabel("Model Group")
        for i, group in enumerate(desired_order):
            points = df[df["group"] == group]["best_epoch"].dropna()
            plt.scatter(points, [i]*len(points), color='k', alpha=0.6, s=30, marker='x', label='_nolegend_')
    elif args.plot_type == "scatter":
        color_map = plt.get_cmap('Set2', len(desired_order))
        for i, g in enumerate(desired_order):
            subset = df[df["group"] == g]
            if "best_caption_score" in subset.columns:
                plt.scatter(subset["best_epoch"], subset["best_caption_score"], label=g, s=80, color=color_map(i))
        plt.xlabel("Best Epoch")
        plt.ylabel("Best Caption Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()


# # Path to your JSONL results
# jsonl_path = r"C:\Users\haganb\Documents\GitHub\MarioDiffusion\best_model_statistics.jsonl"
# output_dir = r"C:\Users\haganb\Documents\GitHub\MarioDiffusion\TEST"

# # Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Load JSONL into a DataFrame
# records = []
# with open(jsonl_path, "r") as f:
#     for line in f:
#         group_stats = json.loads(line)
#         group = group_stats["group"]
#         for model in group_stats["models"]:
#             model["group"] = group
#             records.append(model)
# df = pd.DataFrame(records)

# # Ensure numeric types
# for col in ["best_epoch", "best_val_loss", "best_caption_score"]:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# # 1. Boxplot: best_epoch
# plt.figure(figsize=(10, 10))
# groups = [g for g in desired_order if g in df["group"].unique()]
# data = [df[df["group"] == g]["best_epoch"].dropna() for g in groups]
# plt.boxplot(data)
# plt.xticks(ticks=range(1, len(groups)+1), labels=groups, rotation=45, ha='right')
# plt.xticks(rotation=45, ha='right')
# plt.xlabel("Model Group")
# plt.ylabel("Best Epoch")
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "boxplot_best_epoch_by_group.pdf"))
# plt.close()

# # 1b. Violin plot: best_epoch by group
# plt.figure(figsize=(10, 10))
# parts = plt.violinplot(data, showmeans=True, showmedians=True)
# plt.xticks(range(1, len(groups) + 1), groups, rotation=45, ha='right')
# plt.xlabel("Model Group")
# plt.ylabel("Best Epoch")

# # Make median and mean lines thicker and colored for clarity
# if 'cmedians' in parts:
#     parts['cmedians'].set_color('red')
#     parts['cmedians'].set_linewidth(1.5)
# if 'cmeans' in parts:
#     parts['cmeans'].set_color('blue')
#     parts['cmeans'].set_linewidth(3)

# # Add custom legend for mean and median
# median_line = mlines.Line2D([], [], color='red', linewidth=3, label='Median')
# mean_line = mlines.Line2D([], [], color='blue', linewidth=3, label='Mean')
# plt.legend(handles=[median_line, mean_line], loc='lower left')

# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "violinplot_best_epoch_by_group.pdf"))
# plt.close()

# # 2. Horizontal barplot: mean and median best_epoch by group
# grouped = df.groupby("group")
# means = grouped["best_epoch"].mean()
# groups_reversed = groups[::-1]
# y = range(len(groups_reversed))
# plt.figure(figsize=(10, 10))
# plt.barh(y, means[groups_reversed], height=0.4, label="Mean", align='center', color="skyblue")
# plt.yticks([i + 0.2 for i in y], groups_reversed)

# # Overlay individual data points (strip plot) WITHOUT jitter
# for i, group in enumerate(groups_reversed):
#     points = df[df["group"] == group]["best_epoch"].dropna()
#     plt.scatter(points, [i]*len(points), color='k', alpha=0.6, s=30, marker='x', label='_nolegend_')

# plt.ylabel("Model Group")
# plt.xlabel("Best Epoch")
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "horizontal_barplot_mean_median_best_epoch_by_group.pdf"))
# plt.close()


# # 3. Scatter plot: best_epoch vs best_caption_score, colored by group
# plt.figure(figsize=(10, 10))
# colors = plt.cm.get_cmap('tab20', len(groups))
# for i, g in enumerate(groups):
#     subset = df[df["group"] == g]
#     plt.scatter(subset["best_epoch"], subset["best_caption_score"], label=g, s=80, color=colors(i))
# plt.xlabel("Best Epoch")
# plt.ylabel("Best Caption Score")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "scatter_best_epoch_vs_best_caption_score.pdf"))
# plt.close()


