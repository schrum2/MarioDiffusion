import argparse
from level_dataset import LevelDataset
from tokenizer import Tokenizer 
import json
from captions.caption_generator import GrammarGenerator
from captions.caption_match import compare_captions

def parse_args():
    parser = argparse.ArgumentParser(description="Create validation set of captions")

    parser.add_argument("--seed", type=int, default=512, help="Random seed for reproducibility")

    # Dataset args
    parser.add_argument("--pkl", type=str, default="Mario_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="Mario_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    
    # Output args
    parser.add_argument("--save_file", type=str, default="Mario_ValidationCaptions.json", help="Output file")
    parser.add_argument("--validation_set_size", type=int, default=1000, help="Number of captions for validating generation abilities of model")
    
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

    generator = GrammarGenerator(seed = args.seed, describe_absence=args.describe_absence)

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
