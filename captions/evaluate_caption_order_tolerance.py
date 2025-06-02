import argparse
import itertools
import os
import random
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util.common_settings as common_settings  # adjust import if needed

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
    parser.add_argument("--trials", type=int, default=3, help="Number of times to evaluate each caption permutation")
    parser.add_argument("--inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--game", type=str, choices=["Mario", "LR"], default="Mario", help="Game to evaluate (Mario or Lode Runner)")
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Generate sample
    tokenized = pipe.tokenizer(prompt, return_tensors="pt").to(device)
    sample = pipe.generate(
        **tokenized,
        num_inference_steps=steps,
        guidance_scale=guidance_scale
    )

    scene = sample.squeeze().detach().cpu().numpy().tolist()

    # Assign caption from generated scene
    generated_caption = assign_caption(
        scene,
        id_to_char=id_to_char,
        char_to_id=char_to_id,
        tile_descriptors=tile_descriptors,
        describe_locations=False,
        describe_absence=describe_absence
    )

    # Compare prompt to generated caption
    score = compare_captions(prompt, generated_caption)

    if output:
        print(f"Prompt: {prompt}")
        print(f"Generated caption: {generated_caption}")
        print(f"Score: {score:.4f}")

    return score, sample


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
    pipe.eval()

    # Load tile metadata
    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(args.tileset)

    # Parse caption into phrase permutations
    phrases = [p.strip() for p in args.caption.split('.') if p.strip()]
    permutations = list(itertools.permutations(phrases))

    all_scores = []
    permutation_scores = defaultdict(list)

    for perm in permutations:
        perm_caption = '. '.join(perm) + '.'
        print(f"\nEvaluating permutation: {perm_caption}")
        for trial in range(args.trials):
            score, _ = caption_score_with_assign_and_compare(
                device=device,
                pipe=pipe,
                prompt=perm_caption,
                steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed + trial,
                id_to_char=id_to_char,
                char_to_id=char_to_id,
                tile_descriptors=tile_descriptors,
                describe_absence=False,
                output=False
            )
            permutation_scores[perm_caption].append(score)
            all_scores.append(score)
            print(f"  Trial {trial + 1}: Score = {score:.4f}")

    print("\n--- Summary ---")
    for caption, scores in permutation_scores.items():
        print(f"Permutation: {caption}\n  Average Score: {np.mean(scores):.4f}")

    print(f"\nOverall Average Across All Permutations and Trials: {np.mean(all_scores):.4f}")


if __name__ == "__main__":
    main()
