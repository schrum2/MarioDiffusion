import argparse
import sys
from level_dataset import LevelDataset
import json
from captions.caption_generator import GrammarGenerator
from captions.LR_caption_generator import GrammarGenerator as LR_GrammarGenerator
from captions.caption_match import compare_captions
from captions.LR_caption_match import compare_captions as lr_compare_captions
from captions.MM_caption_generator import GrammarGenerator as MM_GrammarGenerator
from captions.MM_caption_match import compare_captions as mm_compare_captions
from captions.generic_caption_generator import GenericGrammarGenerator

import util.common_settings as common_settings

MM_SIMPLE_TILE_COUNT = common_settings.MM_SIMPLE_TILE_COUNT
MM_FULL_TILE_COUNT = common_settings.MM_FULL_TILE_COUNT
MM_FULL_TILESET = common_settings.MM_FULL_TILESET
MM_SIMPLE_TILESET = common_settings.MM_SIMPLE_TILESET

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
        choices=["Mario", "LR", "MM-Simple", "MM-Full", "MMLV", "MM2", "Other"],
        help="Which game to create a model for (affects sample style and tile count)"
    )
    parser.add_argument(
        "--include_entrance_exit",
        action="store_true",
        default=False,
        help="Include entrance/exit direction phrases in generated Mega Man captions (off by default)"
    )

    # --game MM2 draws each caption from topics learned off the dataset:
    parser.add_argument("--min_topics", type=int, default=1, help="Minimum topics per generated MM2 caption")
    parser.add_argument("--max_topics", type=int, default=8, help="Maximum topics per generated MM2 caption")
    parser.add_argument("--blob_prob", type=float, default=0.5, help="Chance of adding a 'blob of X' phrase to an MM2 entity that already has a count")

    return parser.parse_args()

def main():
    args = parse_args()


    if args.game == "Mario":
        args.num_tiles = common_settings.MARIO_TILE_COUNT
        args.tileset = common_settings.MARIO_TILESET
        generator = GrammarGenerator(
            seed = args.seed, 
            describe_absence=args.describe_absence,
            no_upside_down_pipes=args.no_upside_down_pipes
        )
    elif args.game == "LR":
        args.num_tiles = common_settings.LR_TILE_COUNT
        args.tileset = common_settings.LR_TILESET
        generator = LR_GrammarGenerator(
            seed = args.seed, 
            describe_absence=args.describe_absence,
            no_upside_down_pipes=args.no_upside_down_pipes
        )
    elif args.game == "MM-Simple":
        args.num_tiles = common_settings.MM_SIMPLE_TILE_COUNT
        args.tileset = common_settings.MM_SIMPLE_TILESET
        generator = MM_GrammarGenerator(
            seed=args.seed,
            describe_absence=args.describe_absence,
            include_entrance_exit=args.include_entrance_exit
        )
    elif args.game == "MM-Full":
        args.num_tiles = common_settings.MM_FULL_TILE_COUNT
        args.tileset = common_settings.MM_FULL_TILESET
        generator = MM_GrammarGenerator(
            seed=args.seed,
            describe_absence=args.describe_absence,
            include_entrance_exit=args.include_entrance_exit
        )
    elif args.game == "MMLV":
        args.num_tiles = common_settings.MMLV_TILE_COUNT
        args.tileset = common_settings.MMLV_TILESET
        generator = MM_GrammarGenerator(
            seed=args.seed,
            describe_absence=args.describe_absence,
            include_entrance_exit=args.include_entrance_exit
        )
    elif args.game == "MM2":
        # MM2 topics are learned from the dataset (like "Other" below), so the
        # generator is built after the dataset is loaded.
        args.num_tiles = common_settings.MM2_TILE_COUNT
        args.tileset = common_settings.MM2_TILESET
        generator = None
    elif args.game == "Other":
        # The generic generator learns its topics from the training data, so it is
        # built after the dataset is loaded (below).
        if args.describe_absence:
            print("Error: --describe_absence is not supported with --game Other. "
                  "The generic generator only sees topics that actually occur in the "
                  "data and cannot reliably describe what is absent.")
            sys.exit(1)
        generator = None

# Initialize dataset
    dataset = LevelDataset(
        json_path=args.json,
        tokenizer=None,
        shuffle=True,
        mode="text", # Just captions
        augment=False, # No augmenting just for testing
        num_tiles=args.num_tiles
    )

    # For "Other", learn a generic grammar from the actual training captions.
    if args.game == "Other":
        generator = GenericGrammarGenerator(
            captions=list(dataset),
            seed=args.seed,
        )
        print(generator.describe_topics())
        validation_captions = generate_generic_captions(generator, args)
        save_captions(validation_captions, args.save_file)
        return

    # MM2 likewise learns its topic vocabulary from the dataset's captions. The
    # generator is imported here so other games don't pull in the MM2 tileset.
    if args.game == "MM2":
        from captions.MM2_caption_generator import MM2GrammarGenerator
        generator = MM2GrammarGenerator(
            captions=list(dataset),
            tileset=args.tileset,
            seed=args.seed,
            min_topics=args.min_topics,
            max_topics=args.max_topics,
            blob_prob=args.blob_prob,
        )
        print(generator.describe_topics())
        validation_captions = generate_mm2_captions(generator, args)
        save_captions(validation_captions, args.save_file)
        return

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
            elif args.game in ["MM-Simple", "MM-Full", "MMLV"]:
                compare_score = mm_compare_captions(caption, new_caption)
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

    save_captions(validation_captions, args.save_file)

def generate_generic_captions(generator, args):
    """Generate novel captions with the data-driven generic generator.

    Novelty is checked with the generator's order-/plural-independent signature so
    that no generated caption matches a training caption (or an earlier generated
    one).
    """
    validation_captions = []
    seen_signatures = set(generator.training_signatures)
    # Guard against an infinite loop when the space of novel captions is small.
    max_attempts = max(10000, args.validation_set_size * 100)
    attempts = 0

    while len(validation_captions) < args.validation_set_size and attempts < max_attempts:
        attempts += 1
        new_caption = generator.generate_sentence()
        signature = generator.signature(new_caption)
        if signature in seen_signatures:
            print(f"Discarded duplicate: {new_caption}")
            continue
        seen_signatures.add(signature)
        validation_captions.append(new_caption)

        if len(validation_captions) % 10 == 0:
            print(f"Valid captions so far {len(validation_captions)}")

    if len(validation_captions) < args.validation_set_size:
        print(f"Warning: only generated {len(validation_captions)} unique captions "
              f"after {attempts} attempts (requested {args.validation_set_size}).")

    return validation_captions

def generate_mm2_captions(generator, args):
    """Generate novel MM2 captions by drawing random topic combinations from the
    vocabulary learned off the dataset.

    Deduplicates by exact caption text and caps the number of attempts so a small
    topic vocabulary can't spin forever.
    """
    validation_captions = []
    seen = set()
    max_attempts = max(10000, args.validation_set_size * 100)
    attempts = 0

    while len(validation_captions) < args.validation_set_size and attempts < max_attempts:
        attempts += 1
        new_caption = generator.generate_sentence()
        if not new_caption or new_caption in seen:
            continue
        seen.add(new_caption)
        validation_captions.append(new_caption)

        if len(validation_captions) % 10 == 0:
            print(f"Valid captions so far {len(validation_captions)}")

    if len(validation_captions) < args.validation_set_size:
        print(f"Warning: only generated {len(validation_captions)} unique captions "
              f"after {attempts} attempts (requested {args.validation_set_size}).")

    return validation_captions

def save_captions(validation_captions, save_file):
    new_validation_captions = [{"caption": caption} for caption in validation_captions]
    with open(save_file, 'w') as f:
        json.dump(new_validation_captions, f, indent=4)
    print(f"List successfully saved to '{save_file}'")

if __name__ == "__main__":
    main()