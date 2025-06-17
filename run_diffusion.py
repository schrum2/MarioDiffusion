#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
from level_dataset import visualize_samples, samples_to_scenes
import random
from create_ascii_captions import save_level_data
import util.common_settings as common_settings
from models.pipeline_loader import get_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate levels using a trained diffusion model")
    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of levels to generate")
    parser.add_argument("--output_dir", type=str, default="generated_levels", help="Directory to save generated levels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--inference_steps", type=int, default=common_settings.NUM_INFERENCE_STEPS, help="Number of denoising steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--save_as_json", action="store_true", help="Save generated levels as JSON")
    parser.add_argument("--text_conditional", action="store_true", help="Enable text conditioning")
    parser.add_argument("--level_width", type=int, default=None, help="Overrides width from unet if specified")

    # Hopefully always the user to specify the game they wish to run diffusion on
    parser.add_argument(
        "--game",
        type=str,
        default="Mario",
        choices=["Mario", "LR"],
        help="Which game to create a model for (affects sample style and tile count)"
    )


    # Used to generate captions when generating images
    parser.add_argument("--tileset", default='..\TheVGLC\Super Mario Bros\smb.json', help="Descriptions of individual tile types")
    #parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")

    return parser.parse_args()

def generate_levels(args):
    """Generate level designs using a trained diffusion model"""
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the pipeline
    print(f"Loading model from {args.model_path}...")
    
    pipeline = get_pipeline(args.model_path)

    pipeline.to(device)

    # Determine number of tiles from model
    num_tiles = pipeline.unet.config.in_channels
    print(f"Model configured for {num_tiles} tile types")

    # Get scene size from model config
    if hasattr(pipeline.unet.config, 'sample_size'):
        if isinstance(pipeline.unet.config.sample_size, (tuple, list)):
            scene_height, scene_width = pipeline.unet.config.sample_size
        else:
            scene_height = scene_width = pipeline.unet.config.sample_size
    else:
        raise ValueError("Model config does not have sample_size attribute.")
    if args.level_width is not None:
        scene_width = args.level_width
        print(f"Overriding model width to {scene_width} tiles")
        
    print(f"Model scene size: {scene_height}x{scene_width}")

    # Generate in batches
    total_samples = args.num_samples
    num_batches = (total_samples + args.batch_size - 1) // args.batch_size
    all_samples = []
    
    for batch_idx in range(num_batches):
        # Calculate batch size for this iteration
        current_batch_size = min(args.batch_size, total_samples - batch_idx * args.batch_size)
        print(f"Generating batch {batch_idx+1}/{num_batches} ({current_batch_size} samples)...")
        
        # Generate samples
        with torch.no_grad():
            # Generate samples
            samples = pipeline(
                batch_size=current_batch_size,
                generator=torch.Generator(device).manual_seed(args.seed + batch_idx),
                num_inference_steps=args.inference_steps,
                output_type="tensor",
                height=scene_height,
                width=scene_width,
            ).images

            all_samples.append(samples)

            # Create a unique subdirectory for each batch
            start_index = batch_idx * args.batch_size
            visualize_samples(samples, args.output_dir, True, start_index)
    
    # Concatenate all batches
    all_samples = torch.cat(all_samples, dim=0)[:total_samples]
    print(f"Generated {len(all_samples)} level samples")
    
    # visualizes all samples at once
    # visualize_samples(all_samples, args.output_dir)

    if args.save_as_json:
        scenes = samples_to_scenes(all_samples)
        save_level_data(scenes, args.tileset, os.path.join(args.output_dir, "all_levels.json"), False, args.describe_absence, exclude_broken=False)

if __name__ == "__main__":
    args = parse_args()
    if args.game == "Mario":
        args.num_tiles = common_settings.MARIO_TILE_COUNT
        args.tileset = '..\TheVGLC\Super Mario Bros\smb.json'
    elif args.game == "LR":
        args.num_tiles = common_settings.LR_TILE_COUNT
        args.tileset = '..\TheVGLC\Lode Runner\Loderunner.json'
    else:
        raise ValueError(f"Unknown game: {args.game}")
    generate_levels(args)
