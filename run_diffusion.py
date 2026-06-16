#!/usr/bin/env python
import argparse
import os
import sys
import json
import torch
import numpy as np
from level_dataset import visualize_samples, samples_to_scenes
import random
from create_ascii_captions import save_level_data
import util.common_settings as common_settings
from util.size_utils import dataset_width_range, unet_width_factor, sample_random_width
from models.pipeline_loader import get_pipeline
from LR_create_ascii_captions import save_level_data as lr_save_level_data
from MM_create_ascii_captions import save_level_data as mm_save_level_data


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
    parser.add_argument("--visualize", action="store_true", help="Additionally save each generated sample with its A* path overlaid (filename tagged 'solved'/'unsolved')")
    parser.add_argument("--text_conditional", action="store_true", help="Enable text conditioning")
    parser.add_argument("--level_width", type=int, default=None, help="Overrides width from unet if specified")

    # Randomized width: draw one width per batch within a range, so a single run yields levels
    # of varying sizes. Mutually exclusive in spirit with --level_width (which fixes the width).
    parser.add_argument("--random_width", action="store_true", help="Randomize width per batch within [--min_width, --max_width] (or the range in --width_range_json)")
    parser.add_argument("--min_width", type=int, default=None, help="Min width for --random_width (default: smallest scene width in --width_range_json)")
    parser.add_argument("--max_width", type=int, default=None, help="Max width for --random_width (default: largest scene width in --width_range_json)")
    parser.add_argument("--width_range_json", type=str, default=None, help="Scene-bearing dataset used to derive the --random_width range (defaults to <model_path>/training_widths.json if present)")

    # Hopefully always the user to specify the game they wish to run diffusion on
    parser.add_argument(
        "--game",
        type=str,
        default="Mario",
        choices=["Mario", "LR", "MM-Simple", "MM-Full"],
        help="Which game to create a model for (affects sample style and tile count)"
    )


    # Used to generate captions when generating images
    parser.add_argument("--tileset", default=common_settings.MARIO_TILESET, help="Descriptions of individual tile types")
    #parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")

    return parser.parse_args()

def save_astar_visualizations(all_samples, args):
    """Save an extra copy of each generated sample with its A* path overlaid.

    Images land next to the plain renders in args.output_dir, with the same
    sample_{i} naming plus a 'solved'/'unsolved' tag for traversability."""
    astar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astar")
    if astar_dir not in sys.path:
        sys.path.insert(0, astar_dir)
    from astar_traversability_check import astar_path_image
    from captions.util import extract_tileset

    _, id_to_char, _, tile_descriptors = extract_tileset(args.tileset)
    scenes = samples_to_scenes(all_samples)
    for i, scene in enumerate(scenes):
        try:
            img, solved, stats = astar_path_image(scene, args.game, id_to_char, tile_descriptors)
        except Exception as e:
            print(f"A* visualization failed for sample {i}: {e}")
            continue
        if img is None:
            print(f"Sample {i}: no A* path to draw ({stats})")
            continue
        tag = "solved" if solved else "unsolved"
        out_path = os.path.join(args.output_dir, f"sample_{i} - unconditional - {tag}.png")
        img.save(out_path)
        detail = ", ".join(f"{k}={v}" for k, v in stats.items())
        print(f"Sample {i}: {tag} ({detail}) -> {out_path}")


def resolve_run_width_range(args):
    """Resolve (min_width, max_width) for --random_width.

    Priority: explicit --min_width/--max_width > --width_range_json scenes >
    <model_path>/training_widths.json (written by train_diffusion). Either explicit
    endpoint can also override just one side of a derived range.
    """
    lo, hi = args.min_width, args.max_width
    if lo is not None and hi is not None:
        return lo, hi

    derived = None
    if args.width_range_json:
        derived = dataset_width_range(args.width_range_json)
    if derived is None:
        widths_file = os.path.join(args.model_path, "training_widths.json")
        if os.path.exists(widths_file):
            with open(widths_file) as f:
                info = json.load(f)
            derived = (info["min"], info["max"])
    if derived is None:
        raise ValueError(
            "--random_width could not determine a width range. Provide --min_width and "
            "--max_width, or --width_range_json pointing to a scene-bearing dataset."
        )

    lo = lo if lo is not None else derived[0]
    hi = hi if hi is not None else derived[1]
    return lo, hi


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

    # Set up per-batch random width if requested (one width per batch keeps each batch uniform).
    if args.random_width:
        min_width, max_width = resolve_run_width_range(args)
        width_factor = unet_width_factor(pipeline.unet)
        width_rng = random.Random(args.seed)
        print(f"Randomizing width per batch in [{min_width}, {max_width}] (multiples of {width_factor})")
    else:
        print(f"Model scene size: {scene_height}x{scene_width}")

    # Generate in batches
    total_samples = args.num_samples
    num_batches = (total_samples + args.batch_size - 1) // args.batch_size
    all_samples = []

    for batch_idx in range(num_batches):
        # Calculate batch size for this iteration
        current_batch_size = min(args.batch_size, total_samples - batch_idx * args.batch_size)
        batch_width = sample_random_width(min_width, max_width, width_factor, width_rng) if args.random_width else scene_width
        print(f"Generating batch {batch_idx+1}/{num_batches} ({current_batch_size} samples, width {batch_width})...")

        # Generate samples
        with torch.no_grad():
            # Generate samples
            samples = pipeline(
                batch_size=current_batch_size,
                generator=torch.Generator(device).manual_seed(args.seed + batch_idx),
                num_inference_steps=args.inference_steps,
                output_type="tensor",
                height=scene_height,
                width=batch_width,
            ).images

            all_samples.append(samples)

            # Create a unique subdirectory for each batch
            start_index = batch_idx * args.batch_size
            if args.game == "Mario":
                visualize_samples(samples, args.output_dir, True, start_index)
            elif args.game == "LR":
                visualize_samples(samples, args.output_dir, True, start_index, game='LR')
            elif args.game in ("MM-Simple", "MM-Full"):
                visualize_samples(samples, args.output_dir, True, start_index, game=args.game)
            else:
                raise ValueError(f"Unknown game: {args.game}")
    
    # Concatenate all batches. With --random_width the batches have different widths and
    # can't be concatenated, so flatten to a list of per-sample (C,H,W) tensors instead;
    # save_astar_visualizations / samples_to_scenes iterate either form.
    if args.random_width and len({batch.shape[-1] for batch in all_samples}) > 1:
        all_samples = [sample for batch in all_samples for sample in batch][:total_samples]
    else:
        all_samples = torch.cat(all_samples, dim=0)[:total_samples]
    print(f"Generated {len(all_samples)} level samples")

    # visualizes all samples at once
    # visualize_samples(all_samples, args.output_dir)

    if args.visualize:
        save_astar_visualizations(all_samples, args)

    if args.save_as_json:
        scenes = samples_to_scenes(all_samples)
        if args.game == "Mario":
            save_level_data(scenes, args.tileset, os.path.join(args.output_dir, "all_levels.json"), False, args.describe_absence, exclude_broken=False)
        elif args.game == "LR":
            tileset = common_settings.LR_TILESET
            lr_save_level_data(scenes, tileset, os.path.join(args.output_dir, "all_levels.json"), False, args.describe_absence)
        elif args.game in ("MM-Simple", "MM-Full"):
            mm_save_level_data(scenes, args.tileset, os.path.join(args.output_dir, "all_levels.json"), False, args.describe_absence)

if __name__ == "__main__":
    args = parse_args()
    if args.game == "Mario":
        args.num_tiles = common_settings.MARIO_TILE_COUNT
        args.tileset = common_settings.MARIO_TILESET
    elif args.game == "LR":
        args.num_tiles = common_settings.LR_TILE_COUNT
        args.tileset = common_settings.LR_TILESET
    elif args.game == "MM-Simple":
        args.num_tiles = common_settings.MM_SIMPLE_TILE_COUNT
        args.tileset = common_settings.MM_SIMPLE_TILESET
    elif args.game == "MM-Full":
        args.num_tiles = common_settings.MM_FULL_TILE_COUNT
        args.tileset = common_settings.MM_FULL_TILESET
    else:
        raise ValueError(f"Unknown game: {args.game}")
    generate_levels(args)
