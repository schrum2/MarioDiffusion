import os
import json
import re
from statistics import mean, median
from tqdm import tqdm
import datetime

root = os.getcwd()

# Patterns for directory names
cond_uncond_pattern = re.compile(r"^(Mar1and2-(?:conditional|unconditional)[\w\-]*?)(\d+)$")
mlm_pattern = re.compile(r"^(Mar1and2-MLM[\w\-]*?)(\d+)$")

# Output file names with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_cond_uncond = f"best_model_info_{timestamp}.json"
output_mlm = f"best_mlm_model_info_{timestamp}.json"
output_skipped = f"skipped_model_dirs_{timestamp}.json"

# Data containers
cond_uncond_info = {}
mlm_info = {}
skipped_dirs = []

# Scan all directories in root
for name in tqdm(os.listdir(root), desc="Scanning model directories"):
    full_path = os.path.join(root, name)
    if not os.path.isdir(full_path):
        continue

    # Check which pattern matches
    m_cond = cond_uncond_pattern.match(name)
    m_mlm = mlm_pattern.match(name)

    if m_cond:
        group = m_cond.group(1)
        info_path = os.path.join(full_path, "best_model_info.json")
        if not os.path.exists(info_path):
            skipped_dirs.append({"directory": name, "reason": "missing best_model_info.json"})
            continue
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            best_epoch = info.get("best_epoch")
            best_val_loss = info.get("best_val_loss")
            best_caption_score = info.get("best_caption_score")
            entry = {
                "directory": name,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_caption_score": best_caption_score
            }
            if group not in cond_uncond_info:
                cond_uncond_info[group] = []
            cond_uncond_info[group].append(entry)
        except Exception as e:
            skipped_dirs.append({"directory": name, "reason": f"error reading best_model_info.json: {e}"})

    elif m_mlm:
        group = m_mlm.group(1)
        info_path = os.path.join(full_path, "best_model_info.json")
        if not os.path.exists(info_path):
            skipped_dirs.append({"directory": name, "reason": "missing best_model_info.json"})
            continue
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            best_epoch = info.get("best_epoch")
            best_val_loss = info.get("best_val_loss")
            best_caption_score = info.get("best_caption_score")
            final_epoch_val_loss = info.get("final_epoch_val_loss")
            entry = {
                "directory": name,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_caption_score": best_caption_score,
                "final_epoch_val_loss": final_epoch_val_loss
            }
            if group not in mlm_info:
                mlm_info[group] = []
            mlm_info[group].append(entry)
        except Exception as e:
            skipped_dirs.append({"directory": name, "reason": f"error reading best_model_info.json: {e}"})

    else:
        # Directory does not match any expected pattern
        continue

# Helper function to compute stats
def compute_stats(entries, include_final_epoch_val_loss=False):
    epochs = [e["best_epoch"] for e in entries if e["best_epoch"] is not None]
    val_loss = [e["best_val_loss"] for e in entries if e["best_val_loss"] is not None]
    caption_scores = [e["best_caption_score"] for e in entries if e["best_caption_score"] is not None]
    stats = {
        "total_models_of_this_type": len(entries),
        "epoch_avg": mean(epochs) if epochs else None,
        "epoch_median": median(epochs) if epochs else None,
        "best_val_loss_avg": mean(val_loss) if val_loss else None,
        "best_val_loss_median": median(val_loss) if val_loss else None,
        "caption_score_avg": mean(caption_scores) if caption_scores else None,
        "caption_score_median": median(caption_scores) if caption_scores else None,
        "models": entries
    }
    if include_final_epoch_val_loss:
        final_epoch_val_loss = [e["final_epoch_val_loss"] for e in entries if e.get("final_epoch_val_loss") is not None]
        stats["final_epoch_val_loss_avg"] = mean(final_epoch_val_loss) if final_epoch_val_loss else None
        stats["final_epoch_val_loss_median"] = median(final_epoch_val_loss) if final_epoch_val_loss else None
    return stats

# Write conditional/unconditional stats
with open(output_cond_uncond, "w") as f:
    for group, entries in cond_uncond_info.items():
        group_stats = {"group": group}
        group_stats.update(compute_stats(entries))
        f.write(json.dumps(group_stats) + "\n")

# Write MLM stats
with open(output_mlm, "w") as f:
    for group, entries in mlm_info.items():
        group_stats = {"group": group}
        group_stats.update(compute_stats(entries, include_final_epoch_val_loss=True))
        f.write(json.dumps(group_stats) + "\n")

# Write skipped directories
with open(output_skipped, "w") as f:
    json.dump(skipped_dirs, f, indent=2)

print(f"Done! Wrote:\n  {output_cond_uncond}\n  {output_mlm}\n  {output_skipped}")