import os
import json
import re
from statistics import mean, median
from tqdm import tqdm

root = os.getcwd()

model_info = {}

# Regex: group is everything up to the last digit(s), which are captured as the index
model_dir_pattern = re.compile(r"^(Mar1and2-(?:conditional|unconditional)[\w\-]*?)(\d+)$")

# Only consider directories that start with the desired prefixes
for name in tqdm(os.listdir(root), desc="Scanning model directories"):
    if not (name.startswith("Mar1and2-conditional") or name.startswith("Mar1and2-unconditional")):
        continue
    full_path = os.path.join(root, name)
    if os.path.isdir(full_path):
        m = model_dir_pattern.match(name)
        if m:
            group = m.group(1)
            info_path = os.path.join(full_path, "best_model_info.json")
            if not os.path.exists(info_path):
                continue  # Skip if missing
            with open(info_path, "r") as f:
                info = json.load(f)
            best_epoch = info.get("best_epoch")
            best_val_loss = info.get("best_val_loss")
            best_caption_score = info.get("best_caption_score")
            if group not in model_info:
                model_info[group] = []
            model_info[group].append({
                "directory": name,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_caption_score": best_caption_score
            })

# Write group statistics to JSONL and print as you go
with open("best_model_statistics.jsonl", "w") as stats_file:
    for group, entries in model_info.items():
        epochs = [e["best_epoch"] for e in entries if e["best_epoch"] is not None]
        val_loss = [e["best_val_loss"] for e in entries if e["best_val_loss"] is not None]
        scores = [e["best_caption_score"] for e in entries if e["best_caption_score"] is not None]
        group_stats = {
            "group": group,
            "total_models_of_this_type": len(entries),
            "epoch_avg": mean(epochs) if epochs else None,
            "epoch_median": median(epochs) if epochs else None,
            "best_val_loss_avg": mean(val_loss) if val_loss else None,
            "best_val_loss_median": median(val_loss) if val_loss else None,
            "caption_score_avg": mean(scores) if scores else None,
            "caption_score_median": median(scores) if scores else None,
            "models": entries
        }
        print(json.dumps(group_stats, indent=2))  # Print to console for live feedback
        stats_file.write(json.dumps(group_stats) + "\n")