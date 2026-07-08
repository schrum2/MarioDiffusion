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
# from create_level_json_data import load_tileset, MM2_EXTRA_TILE | Used in Line 196 that is commented out. No need for it.
import util.common_settings as common_settings
from util.size_utils import dataset_width_range, unet_width_factor, sample_random_width
from models.pipeline_loader import get_pipeline
from LR_create_ascii_captions import save_level_data as lr_save_level_data
from MM_create_ascii_captions import save_level_data as mm_save_level_data


def crop_null_border(scene, null_id):
    """Strip whole bottom rows and right columns that are entirely 'null_id'.

    Complete levels are trained with content anchored top-left and the pad region
    (bottom/right) filled with the null/void tile, so trimming full-null bottom rows
    and right columns recovers the level's real extent. At least one row and column
    are always kept. Returns a list of lists.
    """
    grid = np.asarray(scene)
    if grid.ndim != 2 or grid.size == 0:
        return scene
    rows, cols = grid.shape
    while rows > 1 and bool((grid[rows - 1, :cols] == null_id).all()):
        rows -= 1
    while cols > 1 and bool((grid[:rows, cols - 1] == null_id).all()):
        cols -= 1
    return grid[:rows, :cols].tolist()


def resolve_null_tile_id(args):
    """Return the null/void tile id for --crop_null_border: --pad_tile_id if given,
    else the 'null'-descriptor tile resolved from --tileset."""
    if args.pad_tile_id is not None:
        return args.pad_tile_id
    from captions.util import extract_tileset
    _, _, char_to_id, tile_descriptors = extract_tileset(args.tileset)
    null_chars = [c for c, d in tile_descriptors.items() if 'null' in d]
    if not null_chars:
        raise ValueError("--crop_null_border needs a null tile: none found in --tileset; pass --pad_tile_id.")
    return char_to_id[null_chars[0]]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate unconditional level samples using a trained diffusion model")
    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of levels to generate")
    parser.add_argument("--output_dir", type=str, default="generated_levels", help="Directory to save generated levels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--inference_steps", type=int, default=common_settings.NUM_INFERENCE_STEPS, help="Number of denoising steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--save_as_json", action="store_true", help="Save generated levels as JSON")
    parser.add_argument("--visualize", action="store_true", help="Additionally save each generated sample with its A* path overlaid (filename tagged 'solved'/'unsolved')")
    # Jacob: I don't think this text conditional parameter should be necessary in run_diffusion
    parser.add_argument("--text_conditional", action="store_true", help="Enable text conditioning")
    parser.add_argument("--level_width", type=int, default=None, help="Overrides width from unet if specified")
    parser.add_argument("--level_height", type=int, default=None, help="Overrides height from unet if specified (e.g. to request a specific complete-level bucket size)")
    parser.add_argument("--crop_null_border", action="store_true", help="After generation, strip whole rows/columns of the null/void pad tile from the bottom/right of each saved level (recovers the real extent of complete levels trained with null padding)")
    parser.add_argument("--pad_tile_id", type=int, default=None, help="Null/void tile id used by --crop_null_border (defaults to the 'null'-descriptor tile resolved from --tileset)")

    # Randomized width: draw one width per batch within a range, so a single run yields levels
    # of varying sizes. Mutually exclusive in spirit with --level_width (which fixes the width).
    parser.add_argument("--random_width", action="store_true", help="Randomize width per batch within [--min_width, --max_width] (or the range in --width_range_json)")
    parser.add_argument("--min_width", type=int, default=None, help="Min width for --random_width (default: smallest scene width in --width_range_json)")
    parser.add_argument("--max_width", type=int, default=None, help="Max width for --random_width (default: largest scene width in --width_range_json)")
    parser.add_argument("--width_range_json", type=str, default=None, help="Scene-bearing dataset used to derive the --random_width range (defaults to <model_path>/training_widths.json if present)")

    parser.add_argument(
        "--output_format",
        type=str,
        default="ascii",
        choices=["ascii", "image", "both"],
        help="Output format: ascii text files, tile images, or both",
    )

    # Jacob: Maybe we should make it required, but not have a default value?
    # Hopefully always the user to specify the game they wish to run diffusion on
    parser.add_argument(
        "--game",
        type=str,
        default="Mario",
        # Jacob: Still confusing that MM means Mario Maker and not Mega Man
        choices=["Mario", "MM2", "LR", "MM-Simple", "MM-Full", "MMLV"],
        help="Which game to create a model for (affects sample style and tile count)"
    )    

    # Used to generate captions when generating images
    # Jacob: Can we remove this in favor of the --game parameter or is it still useful?
    parser.add_argument("--tileset", default=common_settings.MARIO_TILESET, help="Path to tileset JSON")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Caption mentions when tiles are entirely absent")
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


def samples_to_ascii(samples, id_to_char):
    """Convert [batch, channels, height, width] tensors to lists of ASCII row strings."""
    indices = torch.argmax(samples, dim=1).cpu().numpy()
    results = []
    for grid in indices:
        rows = ["".join(id_to_char.get(int(idx), "?") for idx in row) for row in grid]
        results.append(rows)
    return results


def save_ascii_levels(ascii_levels, output_dir, start_index=0):
    for i, rows in enumerate(ascii_levels):
        path = os.path.join(output_dir, f"level_{start_index + i:04d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows))


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

    # Jacob: This was all from Mario Maker, but I removed it. I don't think it is needed.
    #tile_to_id = load_tileset(args.tileset, extra_tile=MM2_EXTRA_TILE)
    #id_to_char = {v: k for k, v in tile_to_id.items()}
    #print(f"Tileset: {len(tile_to_id)} tile types from {args.tileset}")
    
    # Load the pipeline
    print(f"Loading model from {args.model_path}...")

    pipeline = get_pipeline(args.model_path)

    pipeline.to(device)

    # Jacob: This was all from Mario Maker, but I removed it. I don't think it is needed.
    #        Block embeddings worked in Mario Diffusion without this extra code, so I'm not
    #        sure why we would want/need it. Pretty sure that the pipeline loads block embeddings.

    # When the model was trained with block embeddings the pipeline decodes its
    # own embedding output back to a [batch, num_tiles, H, W] tile distribution
    # internally (see UnconditionalDDPMPipeline.__call__), so the samples reaching
    # us are always in tile space and a plain argmax decodes them. The unet itself
    # has embedding_dim input channels in that case, so compare against the right
    # number to avoid a spurious warning.
    #model_channels = pipeline.unet.config.in_channels
    #uses_block_embeddings = getattr(pipeline, "block_embeddings", None) is not None
    #if not uses_block_embeddings and model_channels != len(tile_to_id):
    #    print(f"Warning: model has {model_channels} channels but tileset has {len(tile_to_id)} tiles")

    #ss = pipeline.unet.config.sample_size
    #if isinstance(ss, (tuple, list)):
    #    scene_height, scene_width = ss
    #else:
    #    scene_height = scene_width = ss

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
    if args.level_height is not None:
        scene_height = args.level_height
        print(f"Overriding model height to {scene_height} tiles")

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
            samples = pipeline(
                batch_size=current_batch_size,
                generator=torch.Generator(device).manual_seed(args.seed + batch_idx),
                num_inference_steps=args.inference_steps,
                output_type="tensor",
                height=scene_height,
                width=batch_width, # Jacob: Changed from scene_width, but I don't think it matters
            ).images

            all_samples.append(samples)

            # Create a unique subdirectory for each batch
            start_index = batch_idx * args.batch_size

        if args.output_format in ("ascii", "both"):
            ascii_levels = samples_to_ascii(samples, id_to_char)
            save_ascii_levels(ascii_levels, args.output_dir, start_index)
            print(f"  Saved {len(ascii_levels)} ASCII levels to {args.output_dir}")

        if args.output_format in ("image", "both"):
            visualize_samples(samples, args.output_dir, True, start_index, game=args.game)
            print(f"  Saved {current_batch_size} level images to {args.output_dir}")
    
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
        if args.crop_null_border:
            null_id = resolve_null_tile_id(args)
            scenes = [crop_null_border(scene, null_id) for scene in scenes]
            print(f"Cropped null border (tile {null_id}) from {len(scenes)} saved level(s)")
        out_path = os.path.join(args.output_dir, "all_levels.json")
        if args.game == "Mario":
            save_level_data(scenes, args.tileset, out_path, False, args.describe_absence, exclude_broken=False)
        elif args.game == "LR":
            tileset = common_settings.LR_TILESET
            lr_save_level_data(scenes, tileset, out_path, False, args.describe_absence)
        elif args.game in ("MM-Simple", "MM-Full", "MMLV"):
            mm_save_level_data(scenes, args.tileset, out_path, False, args.describe_absence)
        if args.game == "MM2": # Jacob: Added for Mario Maker
            # Jacob: This is the code that was in MarioMakerPCG, but I think it just assigned old style Mario captions to the MM levels.
            #        This should be assigning Mario Maker deterministic captions instead.
            save_level_data(scenes, args.tileset, out_path, False, args.describe_absence, exclude_broken=False)
        print(f"Saved {len(scenes)} captioned scenes to {out_path}")

if __name__ == "__main__":
    args = parse_args()
    # Jacob: The if/else staructure below was in MarioDiffusion,
    #        but I would like to find a way to remove it. I don't think
    #        num_tiles is needed at all (it comes from the model) and
    #        we might be able to pass args.game around instead of using
    #        it to define tileset here.
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
    elif args.game == "MMLV":
        args.num_tiles = common_settings.MMLV_TILE_COUNT
        args.tileset = common_settings.MMLV_TILESET
    # Jacob: but in the meantime, Mario Maker deserves its own case
    elif args.game == "MM2":
        args.num_tiles = common_settings.MM2_TILE_COUNT
        args.tileset = common_settings.MM2_TILESET
    else:
        raise ValueError(f"Unknown game: {args.game}")
    generate_levels(args)
