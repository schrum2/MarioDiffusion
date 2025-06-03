import argparse
import itertools
import os
import random
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util.common_settings as common_settings  # adjust import if needed
from level_dataset import LevelDataset, visualize_samples, colors, mario_tiles  # adjust import if needed
from torch.utils.data import DataLoader
from evaluate_caption_adherence import calculate_caption_score_and_samples  # adjust import if needed
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import torch
from tqdm import tqdm

from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from create_ascii_captions import assign_caption
from captions.caption_match import compare_captions  # adjust import if needed
from captions.util import extract_tileset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate caption order tolerance for a diffusion model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--caption", type=str, required=True, help="Caption to evaluate, phrases separated by periods")
    parser.add_argument("--tileset", type=str, help="Path to the tileset JSON file")
    parser.add_argument("--json", type=str, default="datasets\\SMB1_LevelsAndCaptions-regular-train.json", help="Path to dataset json file")
    parser.add_argument("--trials", type=int, default=3, help="Number of times to evaluate each caption permutation")
    parser.add_argument("--inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--game", type=str, choices=["Mario", "LR"], default="Mario", help="Game to evaluate (Mario or Lode Runner)")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    return parser.parse_args()


def setup_environment(seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return device


def caption_score_with_assign_and_compare(
    device,
    pipe,
    prompt,
    steps=25,
    guidance_scale=3.5,
    seed=42,
    id_to_char=None,
    char_to_id=None,
    tile_descriptors=None,
    describe_absence=False,
    output=True
):
    # Load pipeline
    pipe = TextConditionalDDPMPipeline.from_pretrained(args.model_path).to(device)

    # Load tile metadata
    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(args.tileset)

    # TODO: This currently only handles a single caption. Needs to be abel to handle all captions in a dataset.
    #  Separate code below into function, call once per caption in given --json dataset

    # Parse caption into phrase permutations
    phrases = [p.strip() for p in args.caption.split('.') if p.strip()]
    permutations = list(itertools.permutations(phrases))

    # After parsing permutations:
    all_captions = []
    for perm in permutations:
        perm_caption = '. '.join(perm) + '. '
        for trial in range(args.trials):
            all_captions.append(perm_caption)

    all_scores = []

    #print(permutations)

    perm_captions = []
    for perm in permutations:
        perm_captions.append('.'.join(perm) + '.')


    # Create a list of dicts as expected by LevelDataset
    caption_data = [{"scene": None, "caption": cap} for cap in perm_captions]

    # Initialize dataset
    dataset = LevelDataset(
        data_as_list=caption_data,
        shuffle=False,
        mode="text",
        augment=False,
        num_tiles=args.num_tiles,
        negative_captions=False,
        block_embeddings=None
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=len(perm_captions),
        shuffle=False,
        num_workers=4,
        drop_last=False,
        persistent_workers=True
    )


    (avg_score, all_samples, all_prompts) = calculate_caption_score_and_samples(device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, args.describe_absence, output=True, height=common_settings.MARIO_HEIGHT, width=common_settings.MARIO_WIDTH)

    print(f"\nAverage score across all captions: {avg_score:.4f}")
    print("\nAll samples shape:", all_samples.shape)
    print("\nAll prompts:", all_prompts)

    visualizations_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    caption_folder = args.caption.replace(" ", "_").replace(".", "_")
    output_dir = os.path.join(visualizations_dir, caption_folder)

    visualize_samples(
        all_samples,
        output_dir=output_dir,
        prompts=all_prompts[0] if all_prompts else "No prompts available"
    )
    print(f"\nVisualizations saved to: {output_dir}")

    return score, dataset


def permutation_caption_score(
    pipe,
    caption,
    device,
    num_tiles,
    id_to_char,
    char_to_id,
    tile_descriptors,
    inference_steps=25,
    guidance_scale=3.5,
    seed=42,
    describe_absence=False,
    height=None,
    width=None,
    trials=1
):
    # Split caption into phrases and get all permutations
    phrases = [p.strip() for p in caption.split('.') if p.strip()]
    permutations = list(itertools.permutations(phrases))

    # Repeat each permutation for the number of trials
    perm_captions = []
    for perm in permutations:
        perm_caption = '. '.join(perm) + '.'
        for _ in range(trials):
            perm_captions.append(perm_caption)

    # Prepare data for LevelDataset
    caption_data = [{"scene": None, "caption": cap} for cap in perm_captions]

    dataset = LevelDataset(
        data_as_list=caption_data,
        shuffle=False,
        mode="text",
        augment=False,
        num_tiles=num_tiles,
        negative_captions=False,
        block_embeddings=None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=len(perm_captions),
        shuffle=False,
        num_workers=4,
        drop_last=False,
        persistent_workers=True
    )

    avg_score, all_samples, all_prompts = calculate_caption_score_and_samples(
        device, pipe, dataloader, inference_steps, guidance_scale, seed,
        id_to_char, char_to_id, tile_descriptors, describe_absence,
        output=False, height=height, width=width
    )

    return avg_score


def main():
    args = parse_args()
    device = setup_environment(args.seed)

    if args.game == "Mario":
        args.num_tiles = common_settings.MARIO_TILE_COUNT
        args.tileset = '..\TheVGLC\Super Mario Bros\smb.json'
    elif args.game == "LR":
        args.num_tiles = common_settings.LR_TILE_COUNT # TODO
        args.tileset = '..\TheVGLC\Lode Runner\Loderunner.json' # TODO
    else:
        raise ValueError(f"Unknown game: {args.game}")

    # Load pipeline
    pipe = TextConditionalDDPMPipeline.from_pretrained(args.model_path).to(device)

    # Load tile metadata
    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(args.tileset)

    # TODO: This currently only handles a single caption. Needs to be abel to handle all captions in a dataset.
    #  Separate code below into function, call once per caption in given --json dataset

    # Parse caption into phrase permutations
    phrases = [p.strip() for p in args.caption.split('.') if p.strip()]
    permutations = list(itertools.permutations(phrases))

    # After parsing permutations:
    all_captions = []
    for perm in permutations:
        perm_caption = '. '.join(perm) + '. '
        for trial in range(args.trials):
            all_captions.append(perm_caption)

    all_scores = []

    #print(permutations)

    perm_captions = []
    for perm in permutations:
        perm_captions.append('.'.join(perm) + '.')


    # Create a list of dicts as expected by LevelDataset
    caption_data = [{"scene": None, "caption": cap} for cap in perm_captions]

    # Initialize dataset
    dataset = LevelDataset(
        data_as_list=caption_data,
        shuffle=False,
        mode="text",
        augment=False,
        num_tiles=args.num_tiles,
        negative_captions=False,
        block_embeddings=None
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=len(perm_captions),
        shuffle=False,
        num_workers=4,
        drop_last=False,
        persistent_workers=True
    )


    (avg_score, all_samples, all_prompts) = calculate_caption_score_and_samples(device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, args.describe_absence, output=True, height=common_settings.MARIO_HEIGHT, width=common_settings.MARIO_WIDTH)

    print(f"\nAverage score across all captions: {avg_score:.4f}")
    permutation_average = permutation_caption_score(
        pipe,
        args.caption,
        device,
        args.num_tiles,
        id_to_char,
        char_to_id,
        tile_descriptors,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        describe_absence=args.describe_absence,
        height=common_settings.MARIO_HEIGHT,
        width=common_settings.MARIO_WIDTH,
        trials=args.trials
    )
    print("\nPermutation average:", permutation_average)
    print("\nAll samples shape:", all_samples.shape)
    print("\nAll prompts:", all_prompts)

    visualizations_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    caption_folder = args.caption.replace(" ", "_").replace(".", "_")
    output_dir = os.path.join(visualizations_dir, caption_folder)

    visualize_samples(
        all_samples,
        output_dir=output_dir,
        prompts=all_prompts[0] if all_prompts else "No prompts available"
    )
    print(f"\nVisualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
