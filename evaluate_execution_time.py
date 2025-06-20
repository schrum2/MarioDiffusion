import os
import re
import json
import glob

ROOT_DIR = "."
OUTPUT_DIR = os.path.join(ROOT_DIR, "training_runtimes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Find all JSON files ending in -runtime-plus-best.json using glob for robustness
    runtime_jsons = glob.glob(os.path.join(ROOT_DIR, '*-runtime-plus-best.json'))

    groupings = []
    group_lookup = {}
    for runtime_json in runtime_jsons:
        if not os.path.isfile(runtime_json):
            print(f"Skipping {runtime_json}, not a real file.")
            continue

        filename = os.path.basename(runtime_json)
        match = re.match(r"^(.*)-runtime-plus-best\.json$", filename)
        if not match:
            print(f"Skipping {filename}, doesn't match expected pattern.")
            continue

        group_name = match.group(1)
        print(f"Processing {runtime_json} for group: {group_name}")

        try:
            with open(runtime_json, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {runtime_json}, could not open/read: {e}")
            continue

        overall_raw = data.get("overall_runtime", {}).get("raw", {})
        best_epoch_raw = data.get("time_to_best_epoch", {}).get("raw", {})
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
            "time_to_best_epoch": {
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
                "time_to_best_epoch_mean": group["time_to_best_epoch"]["mean"],
                "time_to_best_epoch_individual_times": group["time_to_best_epoch"]["individual_times"]
            })
    # Add only the combined means for the pairs
    for cond, mlm, combined_name in combine_pairs:
        if cond in group_lookup and mlm in group_lookup:
            if len(group_lookup[cond]["overall_runtime"]["individual_times"]) != len(group_lookup[mlm]["overall_runtime"]["individual_times"]):
                print(f"Warning: Individual times length mismatch for {cond} and {mlm}")
            combined_overall_mean = group_lookup[cond]["overall_runtime"]["mean"] + group_lookup[mlm]["overall_runtime"]["mean"]
            combined_overall_individual_times = [
                a + b for a, b in zip(group_lookup[cond]["overall_runtime"]["individual_times"], group_lookup[mlm]["overall_runtime"]["individual_times"])
            ]
            combined_best_mean = group_lookup[cond]["time_to_best_epoch"]["mean"] + group_lookup[mlm]["time_to_best_epoch"]["mean"]
            combined_best_individual_times = [
                a + b for a, b in zip(group_lookup[cond]["time_to_best_epoch"]["individual_times"], group_lookup[mlm]["time_to_best_epoch"]["individual_times"])
            ]
            mean_entries.append({
                "group": combined_name,
                "mlm_overall_runtime_mean": group_lookup[mlm]["overall_runtime"]["mean"],
                "cond_overall_runtime_mean": group_lookup[cond]["overall_runtime"]["mean"],
                "overall_runtime_mean": combined_overall_mean,
                "overall_runtime_individual_times": combined_overall_individual_times,
                "mlm_time_to_best_epoch_mean": group_lookup[mlm]["time_to_best_epoch"]["mean"],
                "cond_time_to_best_epoch_mean": group_lookup[cond]["time_to_best_epoch"]["mean"],
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
