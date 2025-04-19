import argparse
import os
from torch.utils.data import DataLoader
import random
import numpy as np
from level_dataset import LevelDataset, visualize_samples
from tokenizer import Tokenizer 
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Create validation set of captions")
    
    # Dataset args
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="level-diffusion-output", help="Output directory")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer.load(args.pkl)

    # Initialize dataset
    dataset = LevelDataset(
        json_path=args.json,
        tokenizer=tokenizer,
        shuffle=True,
        mode="mlm", # Just captions
        augment=False, # No augmenting just for validation
        num_tiles=args.num_tiles
    )

    for i in range(len(dataset)):
        caption = dataset[i]
        print(caption)


if __name__ == "__main__":
    main()
