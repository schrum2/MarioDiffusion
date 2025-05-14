"""
    This will start as just a script for evolving diffusion models,
    but hopefully it can generalize to GANs too.

    Cite: https://pypi.org/project/cmaes/
"""

import numpy as np
from cmaes import CMA
import argparse
import torch

from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from level_dataset import visualize_samples, convert_to_level_format
from create_ascii_captions import extract_tileset
from create_ascii_captions import assign_caption
import os

def caption_fitness(x):
    """
    Generate the scene, then generate its caption, and compare to a desired caption
    """
    global W, H, C, args
    # Convert x to a scene representation
    latent_input = torch.tensor(x.reshape((1, C, H, W)), dtype=torch.float32)
    # Currently seed matches simulation seed, but a fixed seed could be carried with each genome
    generator = torch.Generator("cuda").manual_seed(args.seed)

    settings = {
        "guidance_scale": args.guidance_scale, 
        "num_inference_steps": args.num_inference_steps,
        "output_type": "tensor",
        "raw_latent_sample": latent_input.to("cuda")
    }
        
    # Include caption if desired
    if True: # Make this a check of whether the model supports text embedding
        settings["caption"] = args.target_caption
        
    images = pipe(
        generator=generator,
        **settings
    ).images

    latent_input.to("cpu")
    # Convert to indices
    sample_indices = convert_to_level_format(images)
        
    # Add level data to the list
    scene = sample_indices[0].tolist() # Always just one scene: (1,16,16)
    global id_to_char, char_to_id, tile_descriptors
    actual_caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, args.describe_absence)

    caption_score = 0

    return caption_score, actual_caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=50, help="Number of generations to run")
    parser.add_argument("--width", type=int, default=16, help="Width of a generated scene")
    parser.add_argument("--height", type=int, default=16, help="Height of a generated scene")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of possible tiles/channels")

    parser.add_argument("--model_path", type=str, required=True, help="Path to model whose latent space will be explored")
    parser.add_argument("--target_caption", type=str, required=True, help="Caption that scenes will be evolved to match")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for diffusion model")
    parser.add_argument("--population_size", type=int, default=10, help="Number of genomes in the population")

    parser.add_argument("--tileset", default=os.path.join('..', 'TheVGLC', 'Super Mario Bros', 'smb.json'), help="Descriptions of individual tile types")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")

    global args
    args = parser.parse_args()

    global W, H, C
    W = args.width
    H = args.height
    C = args.num_tiles

    global pipe
    pipe = TextConditionalDDPMPipeline.from_pretrained(args.model_path).to("cuda")

    global id_to_char, char_to_id, tile_descriptors
    _, id_to_char, char_to_id, tile_descriptors = extract_tileset(args.tileset)

    optimizer = CMA(mean=np.zeros(W*H*C), sigma=1.3, population_size=args.population_size, seed=args.seed)

    for generation in range(args.generations):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value, caption = caption_fitness(x)
            solutions.append((x, value))
            print(f"#{generation} {value}:{caption}")
        optimizer.tell(solutions)