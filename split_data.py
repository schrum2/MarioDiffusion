import argparse
import os
import json
import random
from captions.caption_match import TOPIC_KEYWORDS as MARIO_TOPIC_KEYWORDS
from captions.LR_caption_match import TOPIC_KEYWORDS as LR_TOPIC_KEYWORDS

"""
COMMAND LINE: python split_data.py --json SMB1_LevelsAndCaptions-regular-test.json --train_pct 0.8 --val_pct 0.1 --test_pct 0.1
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Split a levels+captions dataset into train/val/test sets.")
    parser.add_argument("--json", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--game", type=str, required=True, choices=["mario", "loderunner"], help="Game name")
    parser.add_argument("--train_pct", type=float, default=0.8, help="Train split percentage")
    parser.add_argument("--val_pct", type=float, default=0.1, help="Validation split percentage")
    parser.add_argument("--test_pct", type=float, default=0.1, help="Test split percentage")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling")
    return parser.parse_args()

def split_dataset(json_path, train_pct, val_pct, test_pct):
    """Splits the dataset into train/val/test and saves them as new JSON files."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if abs(train_pct + val_pct + test_pct - 1.0) > 1e-6:
        raise ValueError("Train/Val/Test percentages must sum to 1.0")

    random.shuffle(data)

    n = len(data)
    n_train = int(train_pct * n)
    n_val = int(val_pct * n)
    n_test = n - n_train - n_val  # Ensure all samples are used

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

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


def verify_coverage(required_structures):
    """
    Verifies that each split contains the required structures. If a split is missing a required structure,
    swaps entries from other splits to ensure coverage.

    Args:
        dataset (list): The full dataset.
        train_split (list): The training split.
        val_split (list): The validation split.
        test_split (list): The test split.
        required_structures (list): List of required structures to verify.

    Returns:
        tuple: Updated train, validation, and test splits.
    """
    
    # Split the dataset
    train_path, val_path, test_path = split_dataset(args.json, args.train_pct, args.val_pct, args.test_pct)
    
    def check_coverage(split, required_structures):
        """Checks which required structures are present in a split."""
        structure_flags = { # Sets each structure to False initially in the dictionary
            structure: False for structure in required_structures
        }
        for entry in split:
            caption = entry.get("caption", "").lower()
            for structure in required_structures:
                if structure in caption:
                    structure_flags[structure] = True
        return structure_flags

    def find_and_swap(source_split, target_split, missing_structure):
        """Finds an entry with the missing structure in the source split and swaps it with an entry in the target split."""
        for i, entry in enumerate(source_split):
            caption = entry.get("caption", "").lower()
            if missing_structure in caption:
                # Swap the entry
                target_split.append(source_split.pop(i))
                return True
        return False
    
    with open(train_path, 'r') as f:
        train_split = json.load(f)
    with open(val_path, 'r') as f:
        val_split = json.load(f)
    with open(test_path, 'r') as f:
        test_split = json.load(f)

    splits = {"train": train_split, "val": val_split, "test": test_split}

    while True:
        all_covered = True
        for split_name, split in splits.items():
            coverage = check_coverage(split, required_structures)
            for structure, is_present in coverage.items():
                if not is_present:
                    all_covered = False
                    print(f"{split_name} split is missing structure: {structure}")
                    # Find and swap from other splits
                    for other_split_name, other_split in splits.items(): # look at other splits
                        if other_split_name != split_name: # as long as we are not looking at the same splt
                            if find_and_swap(other_split, split, structure):
                                print(f"Swapped {structure} from {other_split_name} to {split_name}")
                                break
        if all_covered:
            break

    return splits["train"], splits["val"], splits["test"]

def upside_down_pipes(dataset):
    """Checks for upside-down pipes in the dataset.
    Returns True if any upside-down pipes are found, False otherwise."""
    for entry in dataset:
        caption = entry.get("caption", "").lower()
        if "upside down pipe" in caption:
            return True
    return False

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    # Choose the correct topic keywords based on the game
    if args.game.lower() == "mario":
        topic_keywords = MARIO_TOPIC_KEYWORDS
    elif args.game.lower() == "loderunner":
        topic_keywords = LR_TOPIC_KEYWORDS
    else:
        raise ValueError("Unsupported game specified")

    required_structures = topic_keywords

    # Only Mario has broken and upside down pipes
    if args.game.lower() == "mario":
        required_structures = [kw for kw in required_structures if "broken" not in kw]
    elif args.game.lower() == "loderunner":
        required_structures = [kw for kw in required_structures if "ceiling" not in kw and "loose block" not in kw]

        with open(args.json, 'r') as f:
            full_dataset = json.load(f)

        if not upside_down_pipes(full_dataset):
            required_structures = [kw for kw in required_structures if "upside down pipe" not in kw]

    train_split, val_split, test_split = verify_coverage(required_structures)