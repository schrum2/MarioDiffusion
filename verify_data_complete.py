import os
import json
import argparse

def count_jsonl_entries(file_path):
    """Count the number of entries in a JSONL file."""
    if not os.path.exists(file_path):
        return None
    
    count = 0
    with open(file_path, 'r') as f:
        for _ in f:
            count += 1
    return count

def verify_json_length(file_path, expected_length):
    """Verify that a JSON file exists and contains a list of expected length."""
    if not os.path.exists(file_path):
        return f"File does not exist: {file_path}"
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                return f"JSON content is not a list in {file_path}"
            if len(data) != expected_length:
                return f"Expected length {expected_length}, but found length {len(data)} in {file_path}"
    except json.JSONDecodeError:
        return f"Invalid JSON format in {file_path}"
    
    return None

def verify_data_completeness(model_path, type_str):
    """Verify all data requirements for a given model path and type."""
    errors = []

    # Check random caption samples
    random_samples = os.path.join(model_path, "samples-from-random-Mar1and2-captions", "all_levels.json")
    error = verify_json_length(random_samples, 100)
    if error:
        errors.append(f"Requirement 1 failed: {error}")

    # Check real caption samples
    real_samples = os.path.join(model_path, "samples-from-real-Mar1and2-captions", "all_levels.json")
    error = verify_json_length(real_samples, 7687)
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

    if errors:
        print("\nVerification failed. The following problems were found:")
        for error in errors:
            print(error)
    else:
        print("\nAll requirements verified successfully!")

def main():
    parser = argparse.ArgumentParser(description="Verify completeness of model evaluation data")
    parser.add_argument("model_path", type=str, help="Path to the model directory")
    parser.add_argument("type", type=str, choices=["absence", "regular"], 
                        help="Type of evaluation (absence or regular)")
    
    args = parser.parse_args()
    verify_data_completeness(args.model_path, args.type)

if __name__ == "__main__":
    main()