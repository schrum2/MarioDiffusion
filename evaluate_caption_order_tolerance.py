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
import json

import numpy as np
import torch
from tqdm import tqdm

from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from captions.util import extract_tileset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate caption order tolerance for a diffusion model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--caption", type=str, required=False, default=None, help="Caption to evaluate, phrases separated by periods")
    parser.add_argument("--tileset", type=str, help="Path to the tileset JSON file")
    parser.add_argument("--json", type=str, default="datasets\\SMB1_LevelsAndCaptions-regular-test.json", help="Path to dataset json file")
    parser.add_argument("--trials", type=int, default=3, help="Number of times to evaluate each caption permutation")
    parser.add_argument("--inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--game", type=str, choices=["Mario", "LR"], default="Mario", help="Game to evaluate (Mario or Lode Runner)")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--save_as_json", action="store_true", help="Save generated levels as JSON")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory if not comparing checkpoints (subdir of model directory)")
    return parser.parse_args()


def setup_environment(seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return device

def permutation_caption_score(
    pipe,
    caption,
    device,
    num_tiles,
    dataloader,
    id_to_char,
    char_to_id,
    tile_descriptors,
    inference_steps=25,
    guidance_scale=3.5,
    seed=42,
    describe_absence=False,
    height=None,
    width=None,
    output=False,
    trials=1,
    max_permutations=10  # Limit the number of permutations to avoid excessive memory usage
):
   

    # Prepare data for LevelDataset
    #caption_data = [{"scene": None, "caption": cap} for cap in perm_captions]

    avg_score, all_samples, all_prompts = calculate_caption_score_and_samples(
        device, pipe, dataloader, inference_steps, guidance_scale, seed,
        id_to_char, char_to_id, tile_descriptors, describe_absence,
        output=output, height=height, width=width
    )

    return avg_score

def permutation_caption_scores_for_data(
    pipe,
    captions,
    device,
    num_tiles,
    dataloader,
    id_to_char,
    char_to_id,
    tile_descriptors,
    inference_steps=25,
    guidance_scale=3.5,
    seed=42,
    describe_absence=False,
    height=None,
    width=None,
    trials=1,
    max_permutations=10  # Limit the number of permutations to avoid excessive memory usage
):
    """
    Compute permutation_caption_score for each caption in captions.
    Returns a list of average scores, one per caption.
    """
    scores = []
    for caption in captions:
        print(f"Evaluating caption: {caption}")
        avg_score = permutation_caption_score(
            pipe=pipe,
            caption=caption,
            device=device,
            num_tiles=num_tiles,
            dataloader=dataloader,
            id_to_char=id_to_char,
            char_to_id=char_to_id,
            tile_descriptors=tile_descriptors,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            describe_absence=describe_absence,
            height=height,
            width=width,
            trials=trials,
            max_permutations=max_permutations
        )
        scores.append(avg_score)
    return scores

def load_captions_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # If the JSON is a list of dicts with a "caption" key
    captions = [entry["caption"] for entry in data if "caption" in entry]
    return captions

def creation_of_parameters(caption, max_permutations=10):
    args = parse_args()
    device = setup_environment(args.seed)

    if args.game == "Mario":
        num_tiles = common_settings.MARIO_TILE_COUNT
        tileset = '..\TheVGLC\Super Mario Bros\smb.json'
    elif args.game == "LR":
        num_tiles = common_settings.LR_TILE_COUNT
        tileset = '..\TheVGLC\Lode Runner\Loderunner.json'
    else:
        raise ValueError(f"Unknown game: {args.game}")

    # Load pipeline
    pipe = TextConditionalDDPMPipeline.from_pretrained(args.model_path).to(device)

    # Load tile metadata
    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset)

    perm_captions = []
    if isinstance(caption, list):
        # captions is a list of caption strings
        phrases_per_caption = [
            [p.strip() for p in cap.split('.') if p.strip()]
            for cap in caption
        ]
        permutations = []
        for phrases in phrases_per_caption:
            perms = list(itertools.permutations(phrases))
            if len(perms) > max_permutations:
                perms = random.sample(perms, max_permutations)
            permutations.append(perms)
        perm_captions = ['.'.join(perm) + '.' for perms in permutations for perm in perms]
    elif isinstance(caption, str):
        # Split caption into phrases and get all permutations
        phrases = [p.strip() for p in caption.split('.') if p.strip()]
        permutations = list(itertools.permutations(phrases))

        for perm in permutations:
            perm_captions.append('.'.join(perm) + '.')



     # Create a list of dicts as expected by LevelDataset
    caption_data = [{"scene": None, "caption": cap} for cap in perm_captions]

    #print("Caption data:", caption_data)

    # Initialize dataset
    dataset = LevelDataset(
        data_as_list=caption_data,
        shuffle=False,
        mode="text",
        augment=False,
        num_tiles=common_settings.MARIO_TILE_COUNT,
        negative_captions=False,
        block_embeddings=None
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=min(16, len(perm_captions)),
        shuffle=False,
        num_workers=4,
        drop_last=False,
        persistent_workers=True
    )


    return pipe, device, id_to_char, char_to_id, tile_descriptors, num_tiles, dataloader, perm_captions

def statistics_of_captions(captions, dataloader, pipe=None, device=None, id_to_char=None, char_to_id=None, tile_descriptors=None, num_tiles=None):
    """
    Calculate statistics of the captions.
    Returns average, standard deviation, minimum, maximum, and median of caption scores.
    """
    args = parse_args()
    if not captions:
        print("No captions found in the provided JSON file.")
        return
    print(f"\nLoaded {len(captions)} captions from {args.json}")

    scores = permutation_caption_scores_for_data(
        pipe,
        captions,
        device,
        num_tiles,
        dataloader,
        id_to_char,
        char_to_id,
        tile_descriptors,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        describe_absence=args.describe_absence,
        height=common_settings.MARIO_HEIGHT,
        width=common_settings.MARIO_WIDTH,
        trials=args.trials,
        max_permutations=10  # Limit the number of permutations to avoid excessive memory usage
    )
    avg_score = np.mean(scores)
    std_dev_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    median_score = np.median(scores)
    
    print("\n-----Scores for each caption permutation-----")
    for i, score in enumerate(scores):
        print(f"Scores for caption {i + 1}:", score)

    print("\n-----Statistics of captions-----")
    print(f"Average score: {avg_score:.4f}")
    print(f"Standard deviation: {std_dev_score:.4f}")
    print(f"Minimum score: {min_score:.4f}")
    print(f"Maximum score: {max_score:.4f}")
    print(f"Median score: {median_score:.4f}")

    return scores, avg_score, std_dev_score, min_score, max_score, median_score

def main():
    args = parse_args()
    if args.caption is None or args.caption == "":
        caption = load_captions_from_json(args.json)
    else:
        caption = args.caption
        #caption = ("many pipes. many coins. , many enemies. many blocks. , many platforms. many question blocks.").split(',')

    pipe, device, id_to_char, char_to_id, tile_descriptors, num_tiles, dataloader, perm_caption = creation_of_parameters(caption, max_permutations=10)
    if not pipe:
        print("Failed to create pipeline.")
        return

    (avg_score, all_samples, all_prompts) = calculate_caption_score_and_samples(device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, args.describe_absence, output=True, height=common_settings.MARIO_HEIGHT, width=common_settings.MARIO_WIDTH)

    #print(f"\nAverage score across all captions: {avg_score:.4f}")

    if args.caption is None or args.caption == "":
        #caption = load_captions_from_json(args.json)
        scores, avg_score, std_dev_score, min_score, max_score, median_score = statistics_of_captions(perm_caption, dataloader, pipe, device, id_to_char, char_to_id, tile_descriptors, num_tiles)

    (avg_score, all_samples, all_prompts) = calculate_caption_score_and_samples(device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, args.describe_absence, output=True, height=common_settings.MARIO_HEIGHT, width=common_settings.MARIO_WIDTH)

    print(f"\nAverage score across all captions: {avg_score:.4f}")

   

    visualizations_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    caption_folder = args.caption.replace(" ", "_").replace(".", "_")
    output_directory = os.path.join(visualizations_dir, caption_folder)

    visualize_samples(
        all_samples,
        output_dir=output_directory,
        prompts=all_prompts[0] if all_prompts else "No prompts available"
    )

    print("\nAll samples shape:", all_samples.shape)
    print("\nAll prompts:", all_prompts)
    print(f"\nVisualizations saved to: {output_directory}")

    if args.caption is None or args.caption == "":
        print(f"\nScores for each caption permutation saved to: {args.save_as_json}")
        # Save results to JSON file
        results = {
            "avg_score": avg_score,
            "all_samples": all_samples.tolist(),  # Convert to list for JSON serialization
            "all_prompts": all_prompts,
            "scores": {
                "scores": scores,
                "num_captions": len(scores),
                "avg": avg_score,
                "std_dev": std_dev_score,
                "min": min_score,
                "max": max_score,
                "median": median_score
            },
        }
    else:
        # Save results for a single caption
        results = {
            "all_samples": all_samples.tolist(),  # Convert to list for JSON serialization
            "avg_score": avg_score,
            "all_prompts": all_prompts,
            "caption": caption
        }    
       

    if args.save_as_json:
        output_json_path = os.path.join(args.output_dir, "evaluation_caption_order_results.json")
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_json_path}")
    else:
        print("Results not saved as JSON file. Use --save_as_json to enable saving.")

if __name__ == "__main__":
    main()
