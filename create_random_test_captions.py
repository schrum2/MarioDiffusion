import argparse
from level_dataset import LevelDataset
import json
from captions.caption_generator import GrammarGenerator
from captions.LR_caption_generator import GrammarGenerator as LR_GrammarGenerator
from captions.caption_match import compare_captions
from captions.LR_caption_match import compare_captions as lr_compare_captions
import util.common_settings as common_settings

def parse_args():
    parser = argparse.ArgumentParser(description="Create random set of captions")

    parser.add_argument("--seed", type=int, default=512, help="Random seed for reproducibility")

    # Dataset args
    parser.add_argument("--json", type=str, default="Mario_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=common_settings.MARIO_TILE_COUNT, help="Number of tile types")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    
    # Output args
    parser.add_argument("--save_file", type=str, default="Mario_RandomCaptions.json", help="Output file")
    parser.add_argument("--validation_set_size", type=int, default=100, help="Number of captions for testing generation abilities of model")

    # Remove upside down pipes from the caption generator
    parser.add_argument("--no_upside_down_pipes", action="store_true", default=False, help="Exclude captions mentioning upside down pipes")
    parser.add_argument(
        "--game",
        type=str,
        default="Mario",
        choices=["Mario", "LR"],
        help="Which game to create a model for (affects sample style and tile count)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    if args.game == "Mario":
        args.num_tiles = common_settings.MARIO_TILE_COUNT
        args.tileset = 'datasets\smb.json'
        generator = GrammarGenerator(
            seed = args.seed, 
            describe_absence=args.describe_absence,
            no_upside_down_pipes=args.no_upside_down_pipes
        )
    elif args.game == "LR":
        args.num_tiles = common_settings.LR_TILE_COUNT
        args.tileset = '..\TheVGLC\Lode Runner\Loderunner.json' 
        generator = LR_GrammarGenerator(
            seed = args.seed, 
            describe_absence=args.describe_absence,
            no_upside_down_pipes=args.no_upside_down_pipes
        )

    # Initialize dataset
    dataset = LevelDataset(
        json_path=args.json,
        tokenizer=None,
        shuffle=True,
        mode="text", # Just captions
        augment=False, # No augmenting just for testing
        num_tiles=args.num_tiles
    )

    validation_captions = []
    while len(validation_captions) < args.validation_set_size:
        new_caption = generator.generate_sentence()
        caption_is_new = True
        # Compare against every caption of original dataset
        for caption in dataset:
            if args.game == "Mario": 
                compare_score = compare_captions(caption, new_caption)
            elif args.game == "LR":
                compare_score = lr_compare_captions(caption, new_caption)
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
