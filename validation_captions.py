import argparse
import os
from torch.utils.data import DataLoader
import random
import numpy as np
from level_dataset import LevelDataset, visualize_samples
from tokenizer import Tokenizer 
import json
from caption_generator import GrammarGenerator
from caption_match import compare_captions

def parse_args():
    parser = argparse.ArgumentParser(description="Create validation set of captions")
    
    # Dataset args
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    
    # Output args
    parser.add_argument("--save_file", type=str, default="SMB1_ValidationCaptions.json", help="Output file")
    parser.add_argument("--validation_set_size", type=int, default=100, help="Number of captions for validating generation abilities of model")
    
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

    #for i in range(len(dataset)):
    #    caption = dataset.get_sample_caption(i)
    #    print(caption)

    generator = GrammarGenerator()

    validation_captions = []
    while len(validation_captions) < args.validation_set_size:
        new_caption = generator.generate_sentence()
        caption_is_new = True
        # Compare against every caption of original dataset
        for i in range(len(dataset)):
            caption = dataset.get_sample_caption(i)
            compare_score = compare_captions(caption, new_caption)
            caption_is_new = compare_score != 1.0 # Perfect score of 1.0 if captions are the same
            if not caption_is_new:
                break

        if not caption_is_new:
            print(f"Discarded duplicate: {new_caption} same as {caption}")
            continue
        else:
            validation_captions.append(new_caption)

        if len(validation_captions) % 10 == 0:
            print(f"Valid captions so far {len(validation_captions)}")

    new_validation_captions = [{"caption": caption} for caption in validation_captions]
    with open(args.save_file, 'w') as f:
        json.dump(new_validation_captions, f, indent=4)
    print(f"List successfully saved to '{args.save_file}'")

if __name__ == "__main__":
    main()
