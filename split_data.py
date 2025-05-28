import argparse
import os
import json
import random

"""
COMMAND LINE: python split_data.py --json SMB1_LevelsAndCaptions-regular-test.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05
"""

def split_dataset(json_path, train_pct, val_pct, test_pct):
    """Splits the dataset into train/val/test and saves them as new JSON files."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    n = len(data)
    train_data, val_data, test_data = [], [], []
    
    # Systematically sample for validation and test sets
    val_interval = int(1 / val_pct)
    test_interval = int(1 / test_pct)
    
    for i, entry in enumerate(data):
        if i % val_interval == 0:
            val_data.append(entry)
        elif i % test_interval == 0:
            test_data.append(entry)
        else:
            train_data.append(entry)
        
    # Save the splits
    base, ext = os.path.splitext(json_path)
    train_path = f"{base}-train{ext}"
    val_path = f"{base}-validate{ext}"
    test_path = f"{base}-test{ext}"
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"Train set saved to: {train_path} ({len(train_data)} samples)")
    print(f"Validation set saved to: {val_path} ({len(val_data)} samples)")
    print(f"Test set saved to: {test_path} ({len(test_data)} samples)")
    return train_path, val_path, test_path

def verify_coverage(dataset, set_name, required_structures):
    """Verifies that the dataset contains the required structures.
    Returns True if all required structures are present, False otherwise."""
    structure_counts = {structure: 0 for structure in required_structures}
    all_required = True

    for entry in dataset:
        caption = entry.get("caption", "").lower()
        for structure in required_structures:
            if structure in caption:
                structure_counts[structure] += 1

    print(f"\nCoverage in {set_name} set:")
    for structure, count in structure_counts.items():
        if count == 0:
            print(f"  WARNING: No {structure} found in {set_name} set!")
            all_required = False # if even one structure is missing, we set this to False
        else:
            print(f"  {structure}: {count} occurrences")
    return all_required

def upside_down_pipes(dataset):
    """Checks for upside-down pipes in the dataset.
    Returns True if any upside-down pipes are found, False otherwise."""
    for entry in dataset:
        caption = entry.get("caption", "").lower()
        if "upside down pipe" in caption:
            return True
    return False



def parse_args():
    parser = argparse.ArgumentParser(description="Split a levels+captions dataset into train/val/test sets.")
    parser.add_argument("--json", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--train_pct", type=float, default=0.9, help="Train split percentage (default 0.9)")
    parser.add_argument("--val_pct", type=float, default=0.05, help="Validation split percentage (default 0.05)")
    parser.add_argument("--test_pct", type=float, default=0.05, help="Test split percentage (default 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    split_dataset(args.json, args.train_pct, args.val_pct, args.test_pct)
    
