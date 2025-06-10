import os
import json
import argparse
from typing import List, Tuple

def count_jsonl_entries(file_path):
    """Count the number of entries in a JSONL file."""
    if not os.path.exists(file_path):
        return None
    
    count = 0
    with open(file_path, 'r') as f:
        for _ in f:
            count += 1
    return count

def verify_json_length(file_path, expected_length, check_prompts=False):
    """Verify that a JSON file exists and contains a list of expected length.
    Optionally verify that the first entry has a non-None prompt field."""
    if not os.path.exists(file_path):
        return f"File does not exist: {file_path}"
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                return f"JSON content is not a list in {file_path}"
            if len(data) != expected_length:
                return f"Expected length {expected_length}, but found length {len(data)} in {file_path}"
            
            if check_prompts and data:
                if "prompt" not in data[0]:
                    return f"First entry missing 'prompt' field in {file_path}"
                if data[0]["prompt"] is None:
                    return f"First entry has None value for 'prompt' in {file_path}"
                
    except json.JSONDecodeError:
        return f"Invalid JSON format in {file_path}"
    
    return None

def verify_data_completeness(model_path, type_str):
    """Verify all data requirements for a given model path and type.
    Returns a list of error messages, or an empty list if verification succeeded."""
    errors = []
    
    # Check random caption samples
    random_samples = os.path.join(model_path, "samples-from-random-Mar1and2-captions", "all_levels.json")
    error = verify_json_length(random_samples, 100, check_prompts=True)
    if error:
        errors.append(f"Requirement 1 failed: {error}")

    # Check real caption samples
    real_samples = os.path.join(model_path, "samples-from-real-Mar1and2-captions", "all_levels.json")
    error = verify_json_length(real_samples, 7687, check_prompts=True)
    if error:
        errors.append(f"Requirement 2 failed: {error}")

    # Check main scores file
    scores_file = os.path.join(model_path, f"Mar1and2_LevelsAndCaptions-{type_str}_scores_by_epoch.jsonl")
    count = count_jsonl_entries(scores_file)
    if count != 27:
        errors.append(f"Requirement 3 failed: Expected 27 entries in {scores_file}, found {count if count is not None else 'file missing'}")

    # Check test scores file
    test_scores_file = os.path.join(model_path, f"Mar1and2_LevelsAndCaptions-{type_str}-test_scores_by_epoch.jsonl")
    count = count_jsonl_entries(test_scores_file)
    if count != 27:
        errors.append(f"Requirement 4 failed: Expected 27 entries in {test_scores_file}, found {count if count is not None else 'file missing'}")

    # Check random test scores file
    random_scores_file = os.path.join(model_path, f"Mar1and2_RandomTest-{type_str}_scores_by_epoch.jsonl")
    count = count_jsonl_entries(random_scores_file)
    if count != 27:
        errors.append(f"Requirement 5 failed: Expected 27 entries in {random_scores_file}, found {count if count is not None else 'file missing'}")

    # Check unconditional samples (long)
    uncond_long = os.path.join(f"{model_path}-unconditional-samples-long", "all_levels.json")
    error = verify_json_length(uncond_long, 100)
    if error:
        errors.append(f"Requirement 6 failed: {error}")

    # Check unconditional samples (short)
    uncond_short = os.path.join(f"{model_path}-unconditional-samples-short", "all_levels.json")
    error = verify_json_length(uncond_short, 100)
    if error:
        errors.append(f"Requirement 7 failed: {error}")
    
    return errors

def find_numbered_directories() -> List[Tuple[str, int, str]]:
    """Find all conditional model directories in current path that end with a number.
    Returns list of tuples (directory_path, number, type).
    Only includes directories containing '-conditional-' and determines type based on name."""
    numbered_dirs = []
    for item in os.listdir('.'):
        if not (os.path.isdir(item) and item[-1].isdigit() and "-conditional-" in item):
            continue
            
        # Get the number at the end of the directory name
        num = ""
        for char in reversed(item):
            if char.isdigit():
                num = char + num
            else:
                break
        
        if num:  # If we found a number
            # Determine type based on directory name
            dir_type = "absence" if "absence" in item.lower() else "regular"
            numbered_dirs.append((item, int(num), dir_type))
            
    return sorted(numbered_dirs, key=lambda x: x[1])  # Sort by number

def main():
    parser = argparse.ArgumentParser(description="Verify completeness of model evaluation data")
    parser.add_argument("--prefix", type=str, help="Prefix of the model directory paths")
    parser.add_argument("--type", type=str, choices=["absence", "regular"], 
                        help="Type of evaluation (absence or regular)")
    parser.add_argument("--start_num", type=int, help="Starting number for model directory range")
    parser.add_argument("--end_num", type=int, help="Ending number for model directory range (inclusive)")
    
    args = parser.parse_args()

    # If any argument is provided, all required arguments must be provided
    if any([args.prefix, args.type, args.start_num, args.end_num]):
        if not all([args.prefix, args.type, args.start_num or args.start_num == 0, args.end_num]):
            parser.error("If any argument is provided, all arguments (--prefix, --type, --start_num, --end_num) are required")
        
        # Parameter-based mode
        for i in range(args.start_num, args.end_num + 1):
            model_path = f"{args.prefix}{i}"
            print(f"\nChecking model directory: {model_path}")
            errors = verify_data_completeness(model_path, args.type)
            if errors:
                print("\nVerification failed. The following problems were found:")
                for error in errors:
                    print(error)
            else:
                print("\nAll requirements verified successfully!")
    else:
        # Automatic discovery mode
        print("Running in automatic directory discovery mode...")
        print("Looking for directories containing '-conditional-' that end in a number...")
        numbered_dirs = find_numbered_directories()
        if not numbered_dirs:
            print("No matching directories found in current directory.")
            return
        
        success_count = 0
        for dir_path, num, dir_type in numbered_dirs:
            print(f"\nChecking directory: {dir_path} (Type: {dir_type})")
            errors = verify_data_completeness(dir_path, dir_type)
            if errors:
                print("Verification failed. Problems found:")
                for error in errors:
                    print(error)
            else:
                print("Verification successful!")
                success_count += 1
        
        print(f"\nVerification complete. {success_count} out of {len(numbered_dirs)} directories passed verification.")

if __name__ == "__main__":
    main()