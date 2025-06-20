import os
import re
import json
import glob

ROOT_DIR = "."
OUTPUT_DIR = os.path.join(ROOT_DIR, "training_runtimes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Find all possible group names by looking for both file types
    plus_best_files = glob.glob(os.path.join(ROOT_DIR, '*-runtime-plus-best.json'))
    old_files = glob.glob(os.path.join(ROOT_DIR, '*-runtime.json'))

    # Extract group names from both file types
    group_names = set()
    for f in plus_best_files:
        m = re.match(r"^(.*)-runtime-plus-best\.json$", os.path.basename(f))
        if m:
            group_names.add(m.group(1))
    for f in old_files:
        m = re.match(r"^(.*)-runtime\.json$", os.path.basename(f))
        if m:
            group_names.add(m.group(1))

    groupings = []
    group_lookup = {}
    for group_name in sorted(group_names):
        plus_best_path = os.path.join(ROOT_DIR, f"{group_name}-runtime-plus-best.json")
        old_path = os.path.join(ROOT_DIR, f"{group_name}-runtime.json")
        data = None
        used_old = False

        if os.path.isfile(plus_best_path):
            with open(plus_best_path, "r") as f:
                data = json.load(f)
        elif os.path.isfile(old_path):
            with open(old_path, "r") as f:
                data = json.load(f)
            used_old = True
        else:
            print(f"Warning: No runtime file found for group {group_name}")
            continue

        if not used_old:
            overall_raw = data.get("overall_runtime", {}).get("raw", {})
            best_epoch_raw = data.get("time_to_best_epoch", {}).get("raw", {})
        else:
            overall_raw = data.get("raw", {})
            best_epoch_raw = None

        group_data = {
            "group": group_name,
            "overall_runtime": {
                "mean": overall_raw.get("mean"),
                "std": overall_raw.get("std"),
                "stderr": overall_raw.get("stderr"),
                "median": overall_raw.get("median"),
                "q1": overall_raw.get("q1"),
                "q3": overall_raw.get("q3"),
                "min": overall_raw.get("min"),
                "max": overall_raw.get("max"),
                "individual_times": overall_raw.get("individual_times", [])
            },
            "time_to_best_epoch": None if used_old else {
                "mean": best_epoch_raw.get("mean"),
                "std": best_epoch_raw.get("std"),
                "stderr": best_epoch_raw.get("stderr"),
                "median": best_epoch_raw.get("median"),
                "q1": best_epoch_raw.get("q1"),
                "q3": best_epoch_raw.get("q3"),
                "min": best_epoch_raw.get("min"),
                "max": best_epoch_raw.get("max"),
                "individual_times": best_epoch_raw.get("individual_times", [])
            }
        }
        groupings.append(group_data)
        group_lookup[group_name] = group_data

    # Write the detailed output JSON
    output_path = os.path.join(OUTPUT_DIR, "all_grouped_runtimes_plus_best.json")
    print(f"all_grouped_runtimes_plus_best.json saved to {output_path}")
    with open(output_path, "w") as f:
        json.dump(groupings, f, indent=2)

    # Prepare the mean summary output
    mean_entries = []
    # Define the pairs to combine
    combine_pairs = [
        ("Mar1and2-conditional-regular", "Mar1and2-MLM-regular", "MLM-regular"),
        ("Mar1and2-conditional-absence", "Mar1and2-MLM-absence", "MLM-absence"),
        ("Mar1and2-conditional-negative", "Mar1and2-MLM-negative", "MLM-negative"),
    ]
    # Track all groups that are part of a combined pair
    combined_groups = set()
    for cond, mlm, _ in combine_pairs:
        combined_groups.add(cond)
        combined_groups.add(mlm)

    # Add all individual means, skipping those in combined pairs
    for group in groupings:
        if group["group"] not in combined_groups:
            mean_entries.append({
                "group": group["group"], 
                "overall_runtime_mean": group["overall_runtime"]["mean"], 
                "overall_runtime_individual_times": group["overall_runtime"]["individual_times"],
                "time_to_best_epoch_mean": group["time_to_best_epoch"]["mean"] if group["time_to_best_epoch"] else None,
                "time_to_best_epoch_individual_times": group["time_to_best_epoch"]["individual_times"] if group["time_to_best_epoch"] else None
            })
    # Add only the combined means for the pairs
    for cond, mlm, combined_name in combine_pairs:
        if cond in group_lookup and mlm in group_lookup:
            cond_times = group_lookup[cond]["overall_runtime"]["individual_times"]
            mlm_times = group_lookup[mlm]["overall_runtime"]["individual_times"]
            if len(cond_times) != len(mlm_times):
                print(f"Warning: Individual times length mismatch for {cond} and {mlm}")
            combined_overall_mean = (
                (group_lookup[cond]["overall_runtime"]["mean"] or 0) +
                (group_lookup[mlm]["overall_runtime"]["mean"] or 0)
            )
            combined_overall_individual_times = [
                a + b for a, b in zip(cond_times, mlm_times)
            ]
            # Handle time_to_best_epoch, which may be None for MLM
            cond_best = group_lookup[cond]["time_to_best_epoch"]
            mlm_best = group_lookup[mlm]["time_to_best_epoch"]
            # If MLM has no time_to_best_epoch, use its overall_runtime for best
            if mlm_best is None:
                mlm_best = group_lookup[mlm]["overall_runtime"]
            if cond_best and mlm_best:
                cond_best_times = cond_best["individual_times"]
                mlm_best_times = mlm_best["individual_times"]
                combined_best_mean = (cond_best["mean"] or 0) + (mlm_best["mean"] or 0)
                combined_best_individual_times = [
                    a + b for a, b in zip(cond_best_times, mlm_best_times)
                ]
            else:
                combined_best_mean = None
                combined_best_individual_times = None
            mean_entries.append({
                "group": combined_name,
                "mlm_overall_runtime_mean": group_lookup[mlm]["overall_runtime"]["mean"],
                "cond_overall_runtime_mean": group_lookup[cond]["overall_runtime"]["mean"],
                "overall_runtime_mean": combined_overall_mean,
                "overall_runtime_individual_times": combined_overall_individual_times,
                "mlm_time_to_best_epoch_mean": mlm_best["mean"] if mlm_best else None,
                "cond_time_to_best_epoch_mean": cond_best["mean"] if cond_best else None,
                "time_to_best_epoch_mean": combined_best_mean,
                "time_to_best_epoch_individual_times": combined_best_individual_times
            })
        else:
            print(f"Warning: Could not find both {cond} and {mlm} for combined mean.")

    # Write the mean summary output JSON
    mean_output_path = os.path.join(OUTPUT_DIR, "mean_grouped_runtimes_plus_best.json")
    print(f"mean_grouped_runtimes_plus_best.json saved to {mean_output_path}")
    with open(mean_output_path, "w") as f:
        json.dump(mean_entries, f, indent=2)

if __name__ == "__main__":
    main()
