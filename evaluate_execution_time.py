import os
import re
import json
import glob

ROOT_DIR = "."
OUTPUT_DIR = os.path.join(ROOT_DIR, "TEST-EXEC-TIME")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Find all JSON files ending in -runtime.json using glob for robustness
    runtime_jsons = glob.glob(os.path.join(ROOT_DIR, '*-runtime.json'))

    groupings = []
    group_lookup = {}
    for runtime_json in runtime_jsons:
        if not os.path.isfile(runtime_json):
            print(f"Skipping {runtime_json}, not a real file.")
            continue

        filename = os.path.basename(runtime_json)
        match = re.match(r"^(.*)-runtime\.json$", filename)
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

        raw_data = data.get("raw", {})
        group_data = {
            "group": group_name,
            "mean": raw_data["mean"],
            "std": raw_data["std"],
            "stderr": raw_data["stderr"],
            "median": raw_data["median"],
            "q1": raw_data["q1"],
            "q3": raw_data["q3"],
            "min": raw_data["min"],
            "max": raw_data["max"],
            "individual_times": raw_data.get("individual_times", [])
        }
        groupings.append(group_data)
        group_lookup[group_name] = group_data

    # Write the detailed output JSON
    output_path = os.path.join(OUTPUT_DIR, "all_grouped_runtimes.json")
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
            mean_entries.append({"group": group["group"], "mean": group["mean"]})
    # Add only the combined means for the pairs
    for cond, mlm, combined_name in combine_pairs:
        if cond in group_lookup and mlm in group_lookup:
            combined_mean = group_lookup[cond]["mean"] + group_lookup[mlm]["mean"]
            mean_entries.append({"group": combined_name,
                                 "mlm_mean": group_lookup[mlm]["mean"],
                                 "cond_mean": group_lookup[cond]["mean"],
                                 "mean": combined_mean})
        else:
            print(f"Warning: Could not find both {cond} and {mlm} for combined mean.")

    # Write the mean summary output JSON
    mean_output_path = os.path.join(OUTPUT_DIR, "mean_grouped_runtimes.json")
    with open(mean_output_path, "w") as f:
        json.dump(mean_entries, f, indent=2)

if __name__ == "__main__":
    main()
