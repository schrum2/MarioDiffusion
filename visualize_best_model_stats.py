import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# Path to your JSONL results
jsonl_path = r"C:\Users\haganb\Documents\GitHub\MarioDiffusion\best_model_statistics.jsonl"
output_dir = r"C:\Users\haganb\Documents\GitHub\MarioDiffusion\TEST"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load JSONL into a DataFrame
records = []
with open(jsonl_path, "r") as f:
    for line in f:
        group_stats = json.loads(line)
        group = group_stats["group"]
        for model in group_stats["models"]:
            model["group"] = group
            records.append(model)
df = pd.DataFrame(records)

# If model is not Mar1and2-unconditional remove prefix "Mar1and2-conditional-"
df["group"] = df["group"].str.replace(r"^Mar1and2-(?:conditional|unconditional)-", "", regex=True)

# Set the desired group order
desired_order = [
    "regular",
    "absence",
    "negative",
    "MiniLM-regular",
    "MiniLM-absence",
    "MiniLM-negative",
    "MiniLMsplit-regular",
    "MiniLMsplit-absence",
    "MiniLMsplit-negative",
    "GTE-regular",
    "GTE-absence",
    "GTE-negative",
    "GTEsplit-regular",
    "GTEsplit-absence",
    "Mar1and2-unconditional"
]
df["group"] = pd.Categorical(df["group"], categories=desired_order, ordered=True)
df = df.sort_values("group")

# Ensure numeric types
for col in ["best_epoch", "best_val_loss", "best_caption_score"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 1. Boxplot: best_epoch
plt.figure(figsize=(12, 6))
groups = [g for g in desired_order if g in df["group"].unique()]
data = [df[df["group"] == g]["best_epoch"].dropna() for g in groups]
plt.boxplot(data, tick_labels=groups)
plt.xticks(rotation=45, ha='right')
plt.title("Distribution of Best Epoch by Model Group")
plt.xlabel("Model Group")
plt.ylabel("Best Epoch")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_best_epoch_by_group.png"))
plt.close()

# 1b. Violin plot: best_epoch by group
plt.figure(figsize=(12, 6))
parts = plt.violinplot(data, showmeans=True, showmedians=True)
plt.xticks(range(1, len(groups) + 1), groups, rotation=45, ha='right')
plt.title("Violin Plot of Best Epoch by Model Group")
plt.xlabel("Model Group")
plt.ylabel("Best Epoch")

# Make median and mean lines thicker and colored for clarity
if 'cmedians' in parts:
    parts['cmedians'].set_color('red')
    parts['cmedians'].set_linewidth(3)
if 'cmeans' in parts:
    parts['cmeans'].set_color('blue')
    parts['cmeans'].set_linewidth(3)

# Add custom legend for mean and median
median_line = mlines.Line2D([], [], color='red', linewidth=3, label='Median')
mean_line = mlines.Line2D([], [], color='blue', linewidth=3, label='Mean')
plt.legend(handles=[median_line, mean_line], loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "violinplot_best_epoch_by_group.png"))
plt.close()

# 2. Barplot: mean and median best_epoch by group
grouped = df.groupby("group")
means = grouped["best_epoch"].mean()
medians = grouped["best_epoch"].median()
x = range(len(groups))
plt.figure(figsize=(12, 6))
plt.bar(x, means[groups], width=0.4, label="Mean", align='center', color="skyblue")
plt.bar([i + 0.4 for i in x], medians[groups], width=0.4, label="Median", align='center', color="orange", alpha=0.7)
plt.xticks([i + 0.2 for i in x], groups, rotation=45, ha='right')
plt.title("Mean and Median Best Epoch by Model Group")
plt.xlabel("Model Group")
plt.ylabel("Best Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "barplot_mean_median_best_best_epoch_by_group.png"))
plt.close()


# 3. Scatter plot: best_epoch vs best_caption_score, colored by group
plt.figure(figsize=(10, 7))
colors = plt.cm.get_cmap('tab20', len(groups))
for i, g in enumerate(groups):
    subset = df[df["group"] == g]
    plt.scatter(subset["best_epoch"], subset["best_caption_score"], label=g, s=80, color=colors(i))
plt.title("Best Epoch vs Best Caption Score")
plt.xlabel("Best Epoch")
plt.ylabel("Best Caption Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_best_epoch_vs_best_caption_score.png"))
plt.close()

# 3b. Pairwise scatterplot matrix (pair plot) for all metrics
pd.plotting.scatter_matrix(
    df[["best_epoch", "best_caption_score"]],
    figsize=(10, 10),
    diagonal='hist',
    alpha=0.7,
    marker='o'
)
plt.suptitle("Pairwise Scatterplot Matrix of Model Metrics")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(output_dir, "scatter_matrix_model_metrics.png"))
plt.close()

# 6. Table: summary statistics (mean, median, std, count) for each group
summary = grouped.agg({
    "best_epoch": ["mean", "median", "std", "count"],
    "best_val_loss": ["mean", "median", "std", "count"],
    "best_caption_score": ["mean", "median", "std", "count"]
})
summary.to_csv(os.path.join(output_dir, "summary_statistics_by_group.csv"))


print("All visualizations and summary table saved to:", output_dir)