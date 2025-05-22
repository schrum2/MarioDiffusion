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
    indices = list(range(n))
    random.shuffle(indices)
    train_end = int(train_pct * n)
    val_end = train_end + int(val_pct * n)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
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