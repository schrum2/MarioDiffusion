import argparse
import os
import json
import random
from captions.caption_match import TOPIC_KEYWORDS as MARIO_TOPIC_KEYWORDS
from captions.LR_caption_match import TOPIC_KEYWORDS as LR_TOPIC_KEYWORDS
from captions.MM_caption_match import TOPIC_KEYWORDS as MM_TOPIC_KEYWORDS


"""
COMMAND LINE: python split_data.py --json_file SMB1_LevelsAndCaptions-regular-test.json --train_pct 0.8 --val_pct 0.1 --test_pct 0.1
"""

# Keys holding deterministically generated captions: "caption" is the legacy single caption,
# "deterministic_captions" the list written by --caption-mode keyed. These are the only captions
# worded from the fixed topic vocabulary, so they are the only ones coverage can be checked
# against; captions under any other key come from an LLM, which words scenes freely.
DETERMINISTIC_CAPTION_KEYS = ("caption", "deterministic_captions")


def caption_text(entry, caption_key):
    """Lowercased text of one entry's captions under caption_key.

    The key holds either a single caption string ("caption") or a list of them
    ("deterministic_captions"); an entry lacking the key contributes no text.
    """
    value = entry.get(caption_key)
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return " ".join(str(item) for item in value).lower()
    return ""


def detect_caption_key(dataset):
    """Which deterministic caption key the dataset uses, or None if it carries neither.

    Datasets captioned in "legacy" mode carry a "caption" string; "keyed" mode stores a list under
    "deterministic_captions" instead. None means the dataset has only LLM-written captions (or no
    captions), which coverage cannot be checked against.
    """
    present = set()
    for entry in dataset:
        if isinstance(entry, dict):
            present.update(entry.keys())
    for key in DETERMINISTIC_CAPTION_KEYS:
        if key in present:
            return key
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Split a levels+captions dataset into train/val/test sets.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--game", type=str, required=True, choices=["MM2", "Mario", "mario", "loderunner", "LR", "mm-simple", "mm-full", "mmlv", "MM-Simple", "MM-Full", "MMLV"], help="Game name")
    parser.add_argument("--train_pct", type=float, default=0.8, help="Train split percentage")
    parser.add_argument("--val_pct", type=float, default=0.1, help="Validation split percentage")
    parser.add_argument("--test_pct", type=float, default=0.1, help="Test split percentage")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling")
    return parser.parse_args()

def split_dataset(json_path, train_pct, val_pct, test_pct):
    """Splits the dataset into train/val/test and saves them as new JSON files."""
    with open(json_path, 'r', encoding='utf-8') as f:
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


def verify_coverage(required_structures, caption_key):
    """
    Verifies that each split contains the required structures. If a split is missing a required structure,
    swaps entries from other splits to ensure coverage.

    Args:
        required_structures (list): List of required structures to verify.
        caption_key (str or None): Deterministic caption key the dataset uses (see
            detect_caption_key). None means there is none to check against, so the check is
            skipped instead of reporting every structure as missing.

    Returns:
        tuple: Updated train, validation, and test splits.
    """
    # Split the dataset
    train_path, val_path, test_path = split_dataset(args.json_file, args.train_pct, args.val_pct, args.test_pct)

    # Without deterministic captions there is no fixed vocabulary to look for, so there is
    # nothing meaningful to enforce and every structure would otherwise report as missing.
    if caption_key is None:
        keys = " or ".join(f"'{key}'" for key in DETERMINISTIC_CAPTION_KEYS)
        print("No deterministic captions found, no coverage check performed.")
        required_structures = []

    
    def check_coverage(split, required_structures):
        """Checks which required structures are present in a split."""
        structure_flags = { # Sets each structure to False initially in the dictionary
            structure: False for structure in required_structures
        }
        for entry in split:
            caption = caption_text(entry, caption_key)
            for structure in required_structures:
                if structure in caption:
                    structure_flags[structure] = True
        return structure_flags

    def find_and_swap(source_split, target_split, missing_structure):
        """Finds an entry with the missing structure in the source split and swaps it with an entry in the target split."""
        for i, entry in enumerate(source_split):
            caption = caption_text(entry, caption_key)
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

    # A required structure can only be guaranteed in every split if the dataset holds at
    # least one entry containing it per split. Drop the ones it can't (e.g. a structure the
    # traversability filter removed entirely) so we warn instead of looping forever trying
    # to cover something that isn't there.
    coverable = []
    for structure in required_structures:
        total = sum(1 for split in splits.values() for entry in split
                    if structure in caption_text(entry, caption_key))
        if total < len(splits):
            where = "absent from the dataset" if total == 0 else f"present in only {total} entries"
            print(f"WARNING: required structure '{structure}' is {where}; it cannot be placed "
                  f"in all {len(splits)} splits, so its coverage is not enforced.")
        else:
            coverable.append(structure)

    # Best-effort balancing: move entries so every split contains each coverable structure.
    # Bounded by a pass budget so structures that co-occur (and keep getting swapped back and
    # forth between splits) can't spin forever; if the budget runs out we proceed as-is.
    max_passes = 100
    for _ in range(max_passes):
        all_covered = True
        for split_name, split in splits.items():
            coverage = check_coverage(split, coverable)
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
    else:
        gaps = {name: [s for s, ok in check_coverage(split, coverable).items() if not ok]
                for name, split in splits.items()}
        gaps = {name: missing for name, missing in gaps.items() if missing}
        print(f"WARNING: could not balance coverage within {max_passes} passes; "
              f"remaining gaps: {gaps}. Proceeding with current splits.")

    return splits["train"], splits["val"], splits["test"]

def upside_down_pipes(dataset, caption_key):
    """Checks for upside-down pipes in the dataset.
    Returns True if any upside-down pipes are found, False otherwise."""
    for entry in dataset:
        caption = caption_text(entry, caption_key)
        if "upside down pipe" in caption:
            return True
    return False

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    with open(args.json_file, 'r', encoding='utf-8') as f:
        full_dataset = json.load(f)
    # Where the deterministic captions live, since --caption-mode keyed stores them under
    # "deterministic_captions" instead of "caption".
    caption_key = detect_caption_key(full_dataset)
    # Choose the correct topic keywords based on the game
    if args.game.lower() == "mario":
        required_structures = MARIO_TOPIC_KEYWORDS
        required_structures = [kw for kw in required_structures if "broken" not in kw]
        if not upside_down_pipes(full_dataset, caption_key):
            required_structures = [kw for kw in required_structures if "upside down pipe" not in kw]
    elif args.game.lower() == "loderunner" or args.game.lower() == "lr":
        required_structures = LR_TOPIC_KEYWORDS
        required_structures = [kw for kw in required_structures if "loose block" not in kw]
        required_structures = [kw for kw in required_structures if "ceiling" not in kw]
    elif args.game.lower() in ["mm-simple", "mm-full", "mmlv"]:
        required_structures = MM_TOPIC_KEYWORDS
    elif args.game.lower() == "mm2":
        # MM2 topics come from the tileset, not a fixed list, so there is nothing to balance
        required_structures = []
    else:
        raise ValueError("Unsupported game specified")
    train_split, val_split, test_split = verify_coverage(required_structures, caption_key)