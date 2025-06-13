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

    # Write the output JSON
    output_path = os.path.join(OUTPUT_DIR, "all_grouped_runtimes.json")
    with open(output_path, "w") as f:
        json.dump(groupings, f, indent=2)

if __name__ == "__main__":
    main()
