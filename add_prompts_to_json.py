import argparse
import json
import os
import shutil

# add_prompts_to_json.py --source_json datasets\\Mar1and2_LevelsAndCaptions-regular.json --target_json Mar1and2-conditional-regular0\\samples-from-real-Mar1and2-captions-OLD\\all_levels.json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_json", required=True, help="Path to the source json with prompts as captions")
    parser.add_argument("--target_json", required=True, help="Json file that gets prompts added")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Open and load the source JSON
    with open(args.source_json, 'r') as f:
        source_data = json.load(f)

    # Open and load the target JSON
    with open(args.target_json, 'r') as f:
        target_data = json.load(f)
        

    if len(source_data) == len(target_data):
        print("Number of source file entries match number of target file entries")
    else:
        print(f"{len(source_data)} entries in source data and {len(target_data)} entries in target data. Cannot add prompts to JSON")
        exit()
        
    # Extract prompts from the source json
    prompts = [entry["caption"] for entry in source_data if "caption" in entry]

    
    # Before we begin, create a copy of all_levels.json and change the name to "all_levels.bak.json"
    # Create a backup of the original target JSON
    backup_path = os.path.splitext(args.target_json)[0] + ".bak.json"
    shutil.copy(args.target_json, backup_path)
    print(f"Backup saved to {backup_path}")
    
    # For each index, insert the prompt into the all_levels json file
    # Insert the prompt into each entry in the target JSON
    for i, prompt in enumerate(prompts):
        target_data[i]["prompt"] = prompt

    # Save the modified target JSON
    with open(args.target_json, 'w') as f:
        json.dump(target_data, f, indent=2)
        print(f"Prompts added and saved to {args.target_json}")


if __name__ == "__main__":
    main()