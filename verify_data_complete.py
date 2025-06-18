import os
import json
import argparse
from typing import List, Tuple
from evaluate_metrics import *
import re
import shutil
from evaluate_solvability import load_scene_caption_data

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
    """Verify that a JSON or JSONL file exists and contains a list/number of expected length.
    Optionally verify that the first entry has a non-None prompt field."""
    if not os.path.exists(file_path):
        return f"File does not exist: {file_path}"

    try:
        if file_path.endswith('astar_result.jsonl'):
            count = 0
            first_entry = None
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        try:
                            first_entry = json.loads(line)
                        except Exception:
                            return f"First line is not valid JSON in {file_path}"
                    count += 1
            if count != expected_length:
                return f"Expected length {expected_length}, but found length {count} in {file_path}"
            if check_prompts and first_entry is not None:
                if "prompt" not in first_entry:
                    return f"First entry missing 'prompt' field in {file_path}"
                if first_entry["prompt"] is None:
                    return f"First entry has None value for 'prompt' in {file_path}"
        elif not file_path.endswith('.jsonl'): # Only json at this point
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
    except Exception as e:
        return f"Error reading {file_path}: {e}"
    return None

def is_zero_iteration(model_path: str) -> bool:
    """Return True if the model path ends with exactly '0' (not -10, -20, etc.)."""
    #return bool(re.search(r'[^0-9]-0$', model_path))
    return bool(re.search(r'[^0-9]0$', model_path))

def is_valid_unconditional_sample(path: str) -> bool:
    pattern = r"unconditional\d+$"
    return bool(re.search(pattern, path.lower()))

def verify_data_completeness(model_path, type_str):
    """Verify all data requirements for a given model path and type.
    Returns a list of error messages, or an empty list if verification succeeded."""
    errors = []
    
    conditional = "-conditional-" in model_path.lower()
    fdm = "fdm" in model_path.lower()
    wgan = "wgan" in model_path.lower() #and "samples" in model_path.lower()
    unconditional = "unconditional" in model_path.lower()
    
    if fdm or conditional:

        # Check random caption samples
        random_samples = os.path.join(model_path, "samples-from-random-Mar1and2-captions", "all_levels.json")
        error = verify_json_length(random_samples, 100, check_prompts=True)
        if error:
            errors.append(f"Requirement 1 failed: {error}")
        # Check if evaluation_metrics.json exists in the same directory as all_levels.json
        evaluation_metrics_path = os.path.join(os.path.dirname(random_samples), "evaluation_metrics.json")
        if not os.path.isfile(evaluation_metrics_path):
            errors.append(f"Requirement 2 failed: 'evaluation_metrics.json' file is missing in {random_samples}.")
        
        if is_zero_iteration(model_path):
            print("ZERO!")
            astar_metrics_path = os.path.join(os.path.dirname(random_samples), "astar_result.jsonl")
            if not os.path.isfile(astar_metrics_path):
                errors.append(f"Requirement 3 failed: 'astar_result.jsonl' file is missing in {random_samples}")
            else:
                error = verify_json_length(astar_metrics_path, 100, check_prompts=False)
                if error:
                    errors.append(f"Requirement 3 failed: {error}")
            astar_metrics_path = os.path.join(os.path.dirname(random_samples), "astar_result_overall_averages.json")
            if not os.path.isfile(astar_metrics_path):
                errors.append(f"Requirement 4 failed: 'astar_result_overall_averages.json' file is missing in {random_samples}")

        # Check real caption samples
        real_samples = os.path.join(model_path, "samples-from-real-Mar1and2-captions", "all_levels.json")
        
        # TODO: Change this to resample to 100 samples. First, this should change the existing evalmetrics file (7867) to have a different name, then it should resample and save to a file called evaluation_metrics.json
        error = verify_json_length(real_samples, 100, check_prompts=True)
        if error:
            errors.append(f"Requirement 5 failed: {error}")
        
        # Check that the all_levels_full file exists and is correct
        real_samples_full = os.path.join(os.path.dirname(real_samples), "all_levels_full.json")
        error = verify_json_length(real_samples_full, 7687, check_prompts=True)
        if error:
            errors.append(f"Requirement 5 failed: {error}")
        
        # Check if evaluation_metrics.json exists in the same directory as all_levels.json
        evaluation_metrics_path = os.path.join(os.path.dirname(real_samples), "evaluation_metrics.json")
        if not os.path.isfile(evaluation_metrics_path):
            errors.append(f"Requirement 6 failed: 'evaluation_metrics.json' file is missing in {real_samples}.")
        
        if is_zero_iteration(model_path):    
            astar_metrics_path = os.path.join(os.path.dirname(real_samples), "astar_result.jsonl")
            if not os.path.isfile(astar_metrics_path):
                errors.append(f"Requirement 7 failed: 'astar_result.jsonl' file is missing in {real_samples}")
            else:
                error = verify_json_length(astar_metrics_path, 100, check_prompts=False)
                if error: 
                    errors.append(f"Requirement 7 failed: {error}")
                
            astar_metrics_path = os.path.join(os.path.dirname(real_samples), "astar_result_overall_averages.json")
            if not os.path.isfile(astar_metrics_path):
                errors.append(f"Requirement 8 failed: 'astar_result_overall_averages.json' file is missing in {real_samples}")

        # Check main scores file
        scores_file = os.path.join(model_path, f"Mar1and2_LevelsAndCaptions-{type_str}_scores_by_epoch.jsonl")
        count = count_jsonl_entries(scores_file)
        if count != 27 and not fdm:
            errors.append(f"Requirement 9 failed: Expected 27 entries in {scores_file}, found {count if count is not None else 'file missing'}")
        elif fdm:
            if count != 11:
                errors.append(f"Requirement 9 failed: Expected 11 entries in {scores_file}, found {count if count is not None else 'file missing'}")

        # Check test scores file
        test_scores_file = os.path.join(model_path, f"Mar1and2_LevelsAndCaptions-{type_str}-test_scores_by_epoch.jsonl")
        count = count_jsonl_entries(test_scores_file)
        if count != 27 and not fdm:
            errors.append(f"Requirement 10 failed: Expected 27 entries in {test_scores_file}, found {count if count is not None else 'file missing'}")
        elif fdm:
            if count != 11:
                errors.append(f"Requirement 10 failed: Expected 11 entries in {test_scores_file}, found {count if count is not None else 'file missing'}")

        # Check random test scores file
        random_scores_file = os.path.join(model_path, f"Mar1and2_RandomTest-{type_str}_scores_by_epoch.jsonl")
        count = count_jsonl_entries(random_scores_file)
        if count != 27 and not fdm:
            errors.append(f"Requirement 11 failed: Expected 27 entries in {random_scores_file}, found {count if count is not None else 'file missing'}")
        elif fdm:
            if count != 11:
                errors.append(f"Requirement 11 failed: Expected 11 entries in {random_scores_file}, found {count if count is not None else 'file missing'}")

        if not fdm:
            # Check unconditional samples (long)
            uncond_long = os.path.join(f"{model_path}-unconditional-samples-long", "all_levels.json")
            error = verify_json_length(uncond_long, 100)
            if error:
                errors.append(f"Requirement 12 failed: {error}")
            # Check if evaluation_metrics.json exists in the same directory as all_levels.json
            evaluation_metrics_path = os.path.join(os.path.dirname(uncond_long), "evaluation_metrics.json")
            if not os.path.isfile(evaluation_metrics_path):
                errors.append(f"Requirement 13 failed: 'evaluation_metrics.json' file is missing in {uncond_long}.")
            
            if is_zero_iteration(model_path):    
                astar_metrics_path = os.path.join(os.path.dirname(uncond_long), "astar_result.jsonl")
                if not os.path.isfile(astar_metrics_path):
                    errors.append(f"Requirement 14 failed: 'astar_result.jsonl' file is missing in {uncond_long}")
                else:
                    error = verify_json_length(astar_metrics_path, 100, check_prompts=False)
                    if error:
                        errors.append(f"Requirement 14 failed: {error}")
                        
                astar_metrics_path = os.path.join(os.path.dirname(uncond_long), "astar_result_overall_averages.json")
                if not os.path.isfile(astar_metrics_path):
                    errors.append(f"Requirement 15 failed: 'astar_result_overall_averages.json' file is missing in {uncond_long}")
            

            # Check unconditional samples (short)
            uncond_short = os.path.join(f"{model_path}-unconditional-samples-short", "all_levels.json")
            error = verify_json_length(uncond_short, 100)
            if error:
                errors.append(f"Requirement 16 failed: {error}")
            # Check if evaluation_metrics.json exists in the same directory as all_levels.json
            evaluation_metrics_path = os.path.join(os.path.dirname(uncond_short), "evaluation_metrics.json")
            if not os.path.isfile(evaluation_metrics_path):
                errors.append(f"Requirement 17 failed: 'evaluation_metrics.json' file is missing in {uncond_short}.")
            
            if is_zero_iteration(model_path):    
                astar_metrics_path = os.path.join(os.path.dirname(uncond_short), "astar_result.jsonl")
                if not os.path.isfile(astar_metrics_path):
                    errors.append(f"Requirement 18 failed: 'astar_result.jsonl' file is missing in {uncond_short}")
                else:
                    error = verify_json_length(astar_metrics_path, 100, check_prompts=False)
                    if error:
                        errors.append(f"Requirement 18 failed: {error}")
                        
                astar_metrics_path = os.path.join(os.path.dirname(uncond_short), "astar_result_overall_averages.json")
                if not os.path.isfile(astar_metrics_path):
                    errors.append(f"Requirement 19 failed: 'astar_result_overall_averages.json' file is missing in {uncond_short}")
                
    elif wgan:
        if not "samples" in model_path.lower(): 
            samples = os.path.join(f"{model_path}-samples", "all_levels.json")
        else:
            samples = os.path.join(model_path, "all_levels.json")
            
        error = verify_json_length(samples, 100)
        if error:
            errors.append(f"Requirement 20 failed: {error}")
        evaluation_metrics_path = os.path.join(os.path.dirname(samples), "evaluation_metrics.json")
        if not os.path.isfile(evaluation_metrics_path):
            errors.append(f"Requirement 21 failed: 'evaluation_metrics'.json file is missing in {samples}")
        astar_metrics_path = os.path.join(os.path.dirname(samples), "astar_result.jsonl")
        
        if is_zero_iteration(model_path):
            print("ZERO! WGAN")
            if not os.path.isfile(astar_metrics_path):
                errors.append(f"Requirement 22 failed: 'astar_result.jsonl' file is missing in {samples}")
            else:
                error = verify_json_length(astar_metrics_path, 100, check_prompts=False)
                print("WGAN 0 ", error)
                if error:
                    errors.append(f"Requirement 22 failed: {error}")
            astar_metrics_path = os.path.join(os.path.dirname(samples), "astar_result_overall_averages.json")
            if not os.path.isfile(astar_metrics_path):
                errors.append(f"Requirement 23 failed: 'astar_result_overall_averages.json' is missing in {samples}")
                
    elif unconditional:
        # Check unconditional-samples-short 
        if not is_valid_unconditional_sample(model_path): 
            raise ValueError(f"Model path {model_path} does not match unconditional sample pattern.")
        
        uncond_short = os.path.join(f"{model_path}-unconditional-samples-short", "all_levels.json")
            
        error = verify_json_length(uncond_short, 100)
        if error:
            errors.append(f"Requirement 24 failed: {error}")
        evaluation_metrics_path = os.path.join(os.path.dirname(uncond_short), "evaluation_metrics.json")
        if not os.path.isfile(evaluation_metrics_path):
            errors.append(f"Requirement 25 failed: 'evaluation_metrics.json' file is missing in {uncond_short}.")
        if is_zero_iteration(model_path):
            astar_metrics_path = os.path.join(os.path.dirname(uncond_short), "astar_result.jsonl")
            if not os.path.isfile(astar_metrics_path):
                errors.append(f"Requirement 26 failed: 'astar_result.jsonl' file is missing in {uncond_short}")
            else:
                error = verify_json_length(astar_metrics_path, 100, check_prompts=False)
                if error:
                    errors.append(f"Requirement 26 failed: {error}")
            
            astar_metrics_path = os.path.join(os.path.dirname(uncond_short), "astar_result_overall_averages.json")
            if not os.path.isfile(astar_metrics_path):
                errors.append(f"Requirement 27 failed: 'astar_result_overall_averages.json' file is missing in {uncond_short}")
                
        
        # Check unconditional-samples-long
        uncond_long = os.path.join(f"{model_path}-unconditional-samples-long", "all_levels.json")
            
        error = verify_json_length(uncond_long, 100)
        if error:
            errors.append(f"Requirement 24 failed: {error}")
        evaluation_metrics_path = os.path.join(os.path.dirname(uncond_long), "evaluation_metrics.json")
        if not os.path.isfile(evaluation_metrics_path):
            errors.append(f"Requirement 25 failed: 'evaluation_metrics.json' file is missing in {uncond_long}.")
        if is_zero_iteration(model_path):
            astar_metrics_path = os.path.join(os.path.dirname(uncond_long), "astar_result.jsonl")
            if not os.path.isfile(astar_metrics_path):
                errors.append(f"Requirement 26 failed: 'astar_result.jsonl' file is missing in {uncond_long}")
            else:
                error = verify_json_length(astar_metrics_path, 100, check_prompts=False)
                if error:
                    errors.append(f"Requirement 26 failed: {error}")
            astar_metrics_path = os.path.join(os.path.dirname(uncond_long), "astar_result_overall_averages.json")
            if not os.path.isfile(astar_metrics_path):
                errors.append(f"Requirement 27 failed: 'astar_result_overall_averages.json' file is missing in {uncond_long}")
        

    return errors


def find_directories_with_prefix(prefix):
    """Find directories that start with the given prefix and do NOT contain 'samples'."""
    return [
        d for d in os.listdir()
        if os.path.isdir(d) and d.startswith(prefix) and "samples" not in d.lower()
    ]

def find_numbered_directories() -> List[Tuple[str, int, str]]:
    """Find all conditional model directories in current path that end with a number.
    Returns list of tuples (directory_path, number, type).
    Only includes directories containing '-conditional-' and determines type based on name."""
    numbered_dirs = []
    for item in os.listdir('.'):
        if not os.path.isdir(item):
            continue
            
        # Match rules
        MarioGPT = "MarioGPT" in item
        is_conditional_with_number = "-conditional-" in item and item[-1].isdigit()
        contains_fdm = "fdm" in item
        #contains_unconditional_number = re.search(r"unconditional\d+-.*samples", item)
        #contains_wgan_number_samples = re.search(r"wgan\d+-samples", item)
        contains_unconditional_number = re.search(r"unconditional\d+$", item)
        contains_wgan_number_samples = re.search(r"wgan\d+$", item)

        if not (MarioGPT or is_conditional_with_number or contains_fdm or contains_unconditional_number or contains_wgan_number_samples):
            continue
        
        
        # Extract trailing number (last numeric sequence at the end)
        num_match = re.search(r"(\d+)(?!.*\d)", item)  # last number in string
        if num_match:
            num = int(num_match.group(1))
            dir_type = "absence" if "absence" in item.lower() else "regular"
            numbered_dirs.append((item, num, dir_type))
        else: # This means we are in the MarioGPT case
            numbered_dirs.append((item, 0, "regular"))
            
    return sorted(numbered_dirs, key=lambda x: x[1])  # Sort by number

def detect_caption_order_tolerance(model_path):
    has_caption_order_tolerance = False
    for file in os.listdir(model_path):
        if "caption_order_tolerance" in file:
                has_caption_order_tolerance = True
                return has_caption_order_tolerance, file
    
    return has_caption_order_tolerance, None

def find_last_line_caption_order_tolerance(model_path, file, key="Caption"):
    file_path = os.path.join(model_path, file)
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
        if not lines:
            return 0
        # Find last line that contains a number
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    for key in data:
                        match = re.match(r"Caption (\d+)", key)
                        if match:
                            return int(match.group(1))
            except ValueError:
                continue
    return 0

def main():
    parser = argparse.ArgumentParser(description="Verify completeness of model evaluation data")
    parser.add_argument("--prefix", type=str, help="Prefix of the model directory paths")
    parser.add_argument("--start_num", type=int, help="Starting number for model directory range")
    parser.add_argument("--end_num", type=int, help="Ending number for model directory range (inclusive)")
    parser.add_argument("--show_successes", default=False, action="store_true", help="Only prints information on complete models")
    parser.add_argument("--show_errors", default=False, action="store_true", help="Only prints information on incomplete models")
    parser.add_argument("--override_metrics", action="store_true", help="Recalculates all metrics if set")
    
    args = parser.parse_args()
    
    # Run default case if all other args are None and debug is either True or False
    arg_values = vars(args)
    non_display_args = [v for k, v in vars(args).items() if k not in {"show_successes", "show_errors", "override_metrics"}]

    
    if all(v is False or v is None for v in non_display_args):
        # Case 1: Automatic discovery mode
        print("Running in automatic directory discovery mode...")
        print("Looking for directories that end in a number...")
        numbered_dirs = find_numbered_directories()
        if not numbered_dirs:
            print("No matching directories found in current directory.")
            return
        
        success_count = 0
        for dir_path, num, dir_type in numbered_dirs:
            print(f"\nChecking directory: {dir_path} (Type: {dir_type})")
            
            # if "MarioGPT" not in dir_path: 
            #     evaluate_metrics(dir_path, "Mar1and2", override=args.override_metrics)
            errors = verify_data_completeness(dir_path, dir_type)
            
            # show_model = (
            #     (errors and not args.show_successes) or
            #     (not errors and not args.show_errors)
            # )

            # if show_model:
            #     print(f"\nChecking directory: {dir_path} (Type: {dir_type})")

            has_caption_order_tolerance, file = detect_caption_order_tolerance(dir_path)
            if has_caption_order_tolerance:
                last_line = find_last_line_caption_order_tolerance(dir_path, file, key="Caption")


            if errors and not args.show_successes:
                print("Verification failed. Problems found:")
                for error in errors:
                    print(error)
            elif not errors and not args.show_errors:
                print("Verification successful!")
                success_count += 1
        
        print(f"\nVerification complete. {success_count} out of {len(numbered_dirs)} directories passed verification.")

    elif args.prefix and args.start_num is None and args.end_num is None:
        # Case 2: Only prefix is provided
        print(f"Scanning all directories with prefix: {args.prefix}")
        matched_dirs = find_directories_with_prefix(args.prefix)
        if not matched_dirs:
            print("No directories found matching the given prefix.")
            return
        
        for model_path in matched_dirs:
            if not args.show_successes and not args.show_errors: print(f"\nChecking model directory: {model_path}")
            dir_type = "absence" if "absence" in model_path.lower() else "regular"

            has_caption_order_tolerance, file = detect_caption_order_tolerance(model_path)

            if has_caption_order_tolerance:
                last_line = find_last_line_caption_order_tolerance(model_path, file, key="Caption")

            errors = verify_data_completeness(model_path, dir_type)
            if errors and not args.show_successes:
                if args.show_errors: print(f"\nChecking directory: {model_path} (Type: {dir_type})")
                print("Verification failed. Problems found:")
                for error in errors:
                    print(error)
            elif not errors and not args.show_errors:
                if args.show_successes: print(f"\nChecking directory: {model_path} (Type: {dir_type})")
                print("Verification successful!")

    elif args.prefix and args.start_num is not None and args.end_num is not None:
        # Case 3: Full manual range mode
        if args.end_num < args.start_num:
            parser.error("--end_num must be greater than or equal to --start_num")

        for i in range(args.start_num, args.end_num + 1):
            model_path = f"{args.prefix}{i}"
            if not args.show_successes and not args.show_errors: print(f"\nChecking model directory: {model_path}")
            dir_type = "absence" if "absence" in model_path.lower() else "regular"

            # Can put check for caption order tolerance here
            has_caption_order_tolerance, file = detect_caption_order_tolerance(model_path)
            if has_caption_order_tolerance:
                last_line = find_last_line_caption_order_tolerance(model_path, file, key="Caption")

            errors = verify_data_completeness(model_path, dir_type)
            if errors and not args.show_successes:
                if args.show_errors: print(f"\nChecking directory: {model_path} (Type: {dir_type})")
                print("Verification failed. Problems found:")
                for error in errors:
                    print(error)
            elif not errors and not args.show_errors:
                if args.show_successes: print(f"\nChecking directory: {model_path} (Type: {dir_type})")
                print("Verification successful!")

    else:
        # Invalid combination
        parser.error("Invalid argument combination. Provide either no arguments, only --prefix, or all three: --prefix, --start_num, and --end_num.")


if __name__ == "__main__":
    main()