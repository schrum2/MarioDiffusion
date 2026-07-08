import argparse
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from datetime import datetime
from level_dataset import LevelDataset, visualize_samples
import json
from models.fdm_pipeline import FDMPipeline
from level_dataset import visualize_samples, convert_to_level_format, samples_to_scenes
from create_ascii_captions import assign_caption, save_level_data
from MM_create_ascii_captions import assign_caption as mm_assign_caption
from captions.MM_caption_match import compare_captions as mm_compare_captions
from LR_create_ascii_captions import assign_caption as lr_assign_caption
from LR_create_ascii_captions import save_level_data as lr_save_level_data
from captions.util import extract_tileset 
from captions.caption_match import compare_captions
from captions.MM2_caption_match import caption_tools as mm2_caption_tools
from captions.LR_caption_match import compare_captions as lr_compare_captions
from tqdm.auto import tqdm
import util.common_settings as common_settings
from util.plotter import Plotter, plot_scores_by_width
from util.size_utils import dataset_width_range, unet_width_factor, sample_random_width
from models.pipeline_loader import get_pipeline
from models.general_training_helper import BucketBatchSampler

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate caption adherence for a pretrained text-conditional diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--game", type=str, default=None, choices=["Mario", "LR", "MM-Simple", "MM-Full", "MMLV", "MM2"], help="Game to evaluate: selects the tileset, scene shape, tile count, and the tiles used for rendering. This is the main way to pick a game, and how Mega Man should be resolved. When omitted, the game is derived from --num_tiles (+ --mm) for backward compatibility.")
    parser.add_argument("--num_tiles", type=int, default=common_settings.MARIO_TILE_COUNT, help="Number of tile types")
    # Jacob: This is confusing to use --mm as Mega Man, because Mario Maker is being added now
    parser.add_argument("--mm", action="store_true", help="Backward-compatible shorthand for Mega Man when --game is not given: routes the 13-tile case to MM-Simple instead of Mario (they share a tile count). Prefer --game MM-Simple / --game MM-Full.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
        
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--all_captions", action="store_true", help="Generate a scene for EVERY stored caption of each entry (caption, caption1, caption2, ...), not just the first. Output entries carry source_index/source_name/caption_index so scenes generated from the same source scene can be grouped. Mario Maker only")
    parser.add_argument("--inference_steps", type=int, default=common_settings.NUM_INFERENCE_STEPS, help="Number of denoising steps") # Large reduction from the 500 used during training
    parser.add_argument("--guidance_scale", type=float, default=common_settings.GUIDANCE_SCALE, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--save_as_json", action="store_true", help="Save generated levels as JSON")
    parser.add_argument("--no_caption_score", action="store_true", help="Skip the caption-adherence score calculation and just generate a sample for each prompt.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted checkpoint comparison run")

    # Used to generate captions when generating images
    parser.add_argument("--tileset", default=common_settings.MARIO_TILESET, help="Descriptions of individual tile types")
    #parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--width", type=int, default=common_settings.MARIO_WIDTH, help="Width of the generated levels")
    parser.add_argument("--height", type=int, default=common_settings.MARIO_HEIGHT, help="Height of the generated levels")

    # Randomized output width (mainly for caption-only sets like RandomTest, where there is
    # no source scene to match). One width is drawn per batch so the batch stays uniform.
    parser.add_argument("--random_width", action="store_true", help="Draw a random width per batch within the training width range instead of using a fixed width")
    parser.add_argument("--min_width", type=int, default=None, help="Min width for --random_width (default: smallest scene width in the resolved range source)")
    parser.add_argument("--max_width", type=int, default=None, help="Max width for --random_width (default: largest scene width in the resolved range source)")
    parser.add_argument("--width_range_json", type=str, default=None, help="Scene-bearing dataset used to derive the --random_width range (typically the training LevelsAndCaptions json)")

    # For scene-bearing datasets (recreating known scenes): generate each caption at its source
    # scene's width. This is applied automatically when the dataset has more than one scene width;
    # the flag forces it on for a single-width dataset too. Batches are bucketed to one width.
    parser.add_argument("--match_scene_width", action="store_true", help="Force generating each caption at its source scene's width even for single-width datasets (auto-enabled for multi-width datasets). Requires scenes; mutually exclusive with --random_width")

    # Output args
    parser.add_argument("--output_dir", type=str, default="text_to_level_results", help="Output directory if not comparing checkpoints (subdir of model directory)")
    parser.add_argument("--save_image_samples", action="store_true", help="Save generated levels in png files")

    parser.add_argument("--compare_checkpoints", action="store_true", default=False, help="Run comparison across all model checkpoints")

    return parser.parse_args()

# Per-game settings: (num_tiles, tileset, height, width). resolve_game() picks a game from --game
# (primary) or, for older calls, from --num_tiles (+ --mm). The tileset must match the tile count
# so the deterministic caption script can read every tile (MM-Full uses the 41-tile MM.json;
# MMLV uses the 43-tile MMLV.json that adds the conveyor tiles).
GAME_SETTINGS = {
    "Mario":     (common_settings.MARIO_TILE_COUNT,     common_settings.MARIO_TILESET,     common_settings.MARIO_HEIGHT,   common_settings.MARIO_WIDTH),
    "LR":        (common_settings.LR_TILE_COUNT,        common_settings.LR_TILESET,        common_settings.LR_HEIGHT,      common_settings.LR_WIDTH),
    "MM-Simple": (common_settings.MM_SIMPLE_TILE_COUNT, common_settings.MM_SIMPLE_TILESET, common_settings.MEGAMAN_HEIGHT, common_settings.MEGAMAN_WIDTH),
    "MM-Full":   (common_settings.MM_FULL_TILE_COUNT,   common_settings.MM_FULL_TILESET,   common_settings.MEGAMAN_HEIGHT, common_settings.MEGAMAN_WIDTH),
    "MMLV":      (common_settings.MMLV_TILE_COUNT,      common_settings.MMLV_TILESET,      common_settings.MEGAMAN_HEIGHT, common_settings.MEGAMAN_WIDTH),
    "MM2":       (common_settings.MM2_TILE_COUNT,       common_settings.MM2_TILESET,       common_settings.MM2_HEIGHT,     common_settings.MM2_WIDTH),
 }

# Jacob: The fact that args.mm is used for Mega Man is confusing and should be fixed.
# Jacob: This should be moved into a util file accessed by many other scripts
def resolve_game(args):
    """Map the CLI args to (game, num_tiles, tileset, height, width, path_to_json).

    --game is the primary selector and the main way to choose Mega Man. When it is omitted, the
    game is derived from --num_tiles (+ --mm) so older calls keep working: 8 -> LR, 41 -> MM-Full,
    43 -> MMLV, 13 -> Mario, or MM-Simple when --mm is set (Mario and MM-Simple share a 13-tile count).
    """
    # Jacob: Mario Maker can be set as the game, but will not fall back as a default
    if args.game is not None:
        game = args.game
    elif args.num_tiles == common_settings.LR_TILE_COUNT:
        game = "LR"
    elif args.num_tiles == common_settings.MMLV_TILE_COUNT:
        game = "MMLV"
    elif args.num_tiles == common_settings.MM_FULL_TILE_COUNT:
        game = "MM-Full"
    elif args.mm: # Mega Man!
        game = "MM-Simple"
    else:
        game = "Mario"
    num_tiles, tileset, height, width = GAME_SETTINGS[game]
    # Lode Runner has always evaluated against its regular caption set, ignoring --json.
    path_to_json = "datasets/LR_LevelsAndCaptions-regular.json" if game == "LR" else args.json
    # Jacob: Should also return the id_to_char and reverse mappings
    return game, num_tiles, tileset, height, width, path_to_json

def resolve_eval_width_range(args):
    """Resolve (min_width, max_width) for --random_width, or None when it is disabled.

    The width range is sourced, in priority order, from:
      1. Explicit --min_width / --max_width (either one can also just override a
         single derived endpoint).
      2. --width_range_json (a scene-bearing dataset).
      3. <model_path>/training_widths.json, written by train_diffusion so the
         BucketBatchSampler's width range follows the model.
      4. The eval --json itself, if it happens to contain scenes.
    """
    if not args.random_width:
        return None

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
    if derived is None and os.path.exists(args.json):
        derived = dataset_width_range(args.json)
    if derived is None:
        raise ValueError(
            "--random_width could not determine a width range. Provide --min_width and "
            "--max_width, or --width_range_json pointing to a scene-bearing dataset."
        )

    lo = lo if lo is not None else derived[0]
    hi = hi if hi is not None else derived[1]
    return lo, hi

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"     # Save within the model path directory

    game, num_tiles, tileset, height, width, path_to_json = resolve_game(args)


    if not args.compare_checkpoints:
        args.output_dir = os.path.join(args.model_path, args.output_dir)
        # Check if output directory already exists
        if os.path.exists(args.output_dir):
            print(f"Error: Output directory '{args.output_dir}' already exists. Please remove it or specify a different output directory.")
            exit(1)
        # Create output directory
        os.makedirs(args.output_dir)

    _, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset)
        
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    # In --compare_checkpoints mode each checkpoint's pipeline is loaded inside
    # track_caption_adherence, so we must not load one from the top-level model dir here:
    # a model whose weights live only in checkpoint subdirs (no top-level "unet") would
    # otherwise crash get_pipeline before the comparison even starts.
    pipe = None
    if not args.compare_checkpoints:
        pipe = get_pipeline(args.model_path).to(device)
        assert(pipe.tokenizer is not None)

    if args.match_scene_width and args.random_width:
        print("Error: --match_scene_width and --random_width are mutually exclusive.")
        exit(1)

    # Jacob: All games have custom captions, so I'm not sure this is the best place to handle Mario Maker.
    # Mario Maker levels get their own captioner and comparison (the SMB code
    # would call every MM2 pipe broken -- the tileset has no <>[] chars).
    mm2 = "mm2" in os.path.basename(tileset).lower()

    # Jacob: No, I think we want to enable this for all games
    #if args.all_captions and not mm2:
    #    print("Error: --all_captions is only supported for Mario Maker datasets.")
    #    exit(1)

    # At least some captions will be from an LLM, and caption adherence can't be computed for those.
    if args.all_captions and args.compare_checkpoints:
        print("Error: --all_captions cannot be combined with --compare_checkpoints.")
        exit(1)

    # Jacob: I feel like it should be possible to generalize this across games by incorporating metadata into the LevelDataset.
    if mm2 and not args.compare_checkpoints:
        # Generate straight from the dataset entries (not a LevelDataset) so each
        # output can be tagged with the source scene its caption came from.
        with open(path_to_json, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        items = expand_mm2_caption_items(raw_data, args.all_captions)
        if not items:
            print(f"Error: no captions found in {path_to_json}")
            exit(1)
        print(f"Generating {len(items)} scenes from {len(raw_data)} dataset entries...")
        avg_score, results = mm2_caption_adherence(args, device, pipe, items, tileset)
        print(f"Average caption adherence score: {avg_score:.4f}")
        if args.save_as_json:
            out_path = os.path.join(args.output_dir, "all_levels.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved {len(results)} captioned scenes to {out_path}")
        return

    # Load once. LevelDataset.data holds the raw entries (scenes included) regardless of mode,
    # so we can inspect the set of scene widths here to decide how to generate.
    dataset = LevelDataset(
        json_path=path_to_json,
        tokenizer=None,
        shuffle=False,
        mode="text",
        augment=False,
        # Jacob: I think args.num_tiles was used to allow overriding the default with any size,
        #        but using num_tiles from resolve_game is better in general. However, we might revert to args.num_tiles
        num_tiles=num_tiles,
        #num_tiles=args.num_tiles
    )
    scene_widths = {len(item["scene"][0]) for item in dataset.data if isinstance(item, dict) and item.get("scene")}

    if args.match_scene_width and not scene_widths:
        print(f"Error: --match_scene_width requires a scene-bearing dataset, but '{path_to_json}' has caption-only entries.")
        exit(1)

    # Datasets with more than one scene shape default to recreating each caption at its source
    # scene's width. Homogeneous datasets (one width) and caption-only sets are left on the old
    # fixed-width path. --match_scene_width forces it on; --random_width opts out.
    if len(scene_widths) > 1 and not args.random_width and not args.match_scene_width:
        print(f"Detected {len(scene_widths)} scene widths {sorted(scene_widths)} in {os.path.basename(path_to_json)}; matching generation width to each source scene.")
        args.match_scene_width = True

    # --match_scene_width needs the scenes, so switch to diff_text mode and bucket batches by
    # width (each batch must be a single width). Otherwise captions-only "text" mode is enough.
    if args.match_scene_width:
        dataset.mode = "diff_text"
        dataloader = DataLoader(
            dataset,
            batch_sampler=BucketBatchSampler(dataset, args.batch_size, drop_last=False, shuffle=False),
            num_workers=4
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )

    if args.compare_checkpoints:
        scores_by_epoch = track_caption_adherence(args, device, dataloader, id_to_char, char_to_id, tile_descriptors)

    else:
        # Just run on one model and get samples as well
        width_range = resolve_eval_width_range(args)
        per_width_scores = {}
        avg_score, all_samples, all_prompts, _ = calculate_caption_score_and_samples(device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, args.describe_absence, output=False, height=height, width=width, random_width=args.random_width, width_range=width_range, match_scene_width=args.match_scene_width, per_width_scores=per_width_scores, compute_score=not args.no_caption_score, game=game)

        if avg_score is not None:
            print(f"Average caption adherence score: {avg_score:.4f}")
        print(f"Generated {len(all_samples)} level samples")
        # Show how many samples were generated at each width and how well each width scored.
        # A width missing here means no caption was generated at that size.
        if per_width_scores:
            print("Samples and caption adherence by scene width:")
            for w in sorted(per_width_scores):
                scores = per_width_scores[w]
                print(f"\twidth {w}: {len(scores)} samples, avg score {sum(scores) / len(scores):.4f}")
        
        if args.save_image_samples:
            if isinstance(all_samples, list):
                # Mixed widths can't be stacked into one tensor; render each sample on its own.
                for i, sample in enumerate(all_samples):
                    visualize_samples(sample.unsqueeze(0), args.output_dir, start_index=i, prompts=[all_prompts[i]], game=game)
            else:
                visualize_samples(all_samples, args.output_dir, prompts=all_prompts, game=game)

        if args.save_as_json:
            scenes = samples_to_scenes(all_samples)
            if args.no_caption_score:
                # These prompts are LLM/natural-language, so no adherence score is computed. Keep the
                # original input in "prompt", and also attach a deterministic "caption" derived from
                # the generated scene by the per-game caption script. The "caption" field is what lets
                # the json open in ascii_data_browser.py (which reads sample["caption"]).
                paired = []
                for prompt, scene in zip(all_prompts, scenes):
                    if game == "LR":
                        capt_scene = [[tile % common_settings.LR_TILE_COUNT for tile in row] for row in scene]
                        caption = lr_assign_caption(capt_scene, id_to_char, char_to_id, tile_descriptors, False, args.describe_absence)
                    elif game in ("MM-Simple", "MM-Full", "MMLV"):
                        caption = mm_assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, args.describe_absence)
                    else:  # Mario
                        caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, args.describe_absence)
                    paired.append({"prompt": prompt, "caption": caption, "scene": scene})
                with open(os.path.join(args.output_dir, "all_levels.json"), "w") as f:
                    json.dump(paired, f, indent=4)
            # Jacob: I'm not certain that the same function call here will work for both Mario and MM2 (Mario Maker)
            elif game == "Mario" or game == "MM2":
                save_level_data(scenes, args.tileset, os.path.join(args.output_dir, "all_levels.json"), False, args.describe_absence, exclude_broken=False, prompts=all_prompts)
            elif game in ("MM-Simple", "MM-Full", "MMLV"):
                # Same output shape as the no_caption_score MM2 path: keep the input prompt and a
                # deterministic caption derived from the generated scene so the json opens in
                # ascii_data_browser.py. Without this branch MM2 writes nothing when scoring is on.
                paired = []
                for prompt, scene in zip(all_prompts, scenes):
                    caption = mm_assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, args.describe_absence)
                    paired.append({"prompt": prompt, "caption": caption, "scene": scene})
                with open(os.path.join(args.output_dir, "all_levels.json"), "w") as f:
                    json.dump(paired, f, indent=4)
            elif game == "LR":
                scenes = [
                            [[tile % common_settings.LR_TILE_COUNT for tile in row] for row in scene]
                            for scene in scenes
                        ]
                lr_save_level_data(scenes, tileset, os.path.join(args.output_dir, "all_levels.json"), False, args.describe_absence)


def expand_mm2_caption_items(data, all_captions):
    """One generation item per caption, each tagged with its source entry.
    With all_captions, every stored caption of an entry ("caption", "caption1",
    ...) becomes an item, so downstream metrics can compare the scenes those
    sibling captions produce. Entries can be caption-only (no scene)."""
    items = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        captions = [entry["caption"]] if entry.get("caption") else []
        if all_captions:
            i = 1
            while f"caption{i}" in entry:
                captions.append(entry[f"caption{i}"])
                i += 1
        for cidx, cap in enumerate(captions):
            items.append({
                "caption": cap,
                "scene": entry.get("scene"),
                "source_index": idx,
                "source_name": entry.get("name"),
                "caption_index": cidx,
            })
    return items


def mm2_caption_adherence(args, device, pipe, items, tileset):
    """Generate one scene per item and score it with the MM2 captioner. Batches
    are bucketed by scene size (an item's source scene shape, else
    args.height x args.width) since a batch must share one shape. Returns
    (average score, results); each result carries the prompt, scene, its
    deterministic caption, the score, and the source metadata."""
    assign_caption_fn, compare_captions_fn = mm2_caption_tools(tileset)

    by_shape = {}
    for item in items:
        if item.get("scene"):
            shape = (len(item["scene"]), len(item["scene"][0]))
        else:
            shape = (args.height, args.width)
        by_shape.setdefault(shape, []).append(item)

    results = []
    score_sum = 0.0
    per_shape_scores = {}
    for shape in sorted(by_shape):
        height, width = shape
        bucket = by_shape[shape]
        for start in tqdm(range(0, len(bucket), args.batch_size),
                          desc=f"Generating {height}x{width}", unit="batch"):
            batch = bucket[start:start + args.batch_size]
            captions = [item["caption"] for item in batch]
            generator = torch.Generator(device).manual_seed(int(args.seed))
            with torch.no_grad():
                samples = pipe(
                    caption=captions,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    guidance_scale=args.guidance_scale,
                    output_type="tensor",
                    batch_size=len(captions),
                    generator=generator,
                ).images

            if args.save_image_samples:
                visualize_samples(samples, args.output_dir, start_index=len(results),
                                  prompts=captions, game="MM2")

            scenes = samples_to_scenes(samples)
            for item, scene in zip(batch, scenes):
                actual_caption = assign_caption_fn(scene)
                score = compare_captions_fn(item["caption"], actual_caption)
                score_sum += score
                per_shape_scores.setdefault(shape, []).append(score)
                results.append({
                    "prompt": item["caption"],
                    "scene": scene,
                    "caption": actual_caption,
                    "score": score,
                    "source_index": item["source_index"],
                    "source_name": item["source_name"],
                    "caption_index": item["caption_index"],
                })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(per_shape_scores) > 1:
        print("Samples and caption adherence by scene size:")
        for shape in sorted(per_shape_scores):
            scores = per_shape_scores[shape]
            print(f"\t{shape[0]}x{shape[1]}: {len(scores)} samples, avg score {sum(scores) / len(scores):.4f}")

    return score_sum / len(results), results


def track_caption_adherence(args, device, dataloader, id_to_char, char_to_id, tile_descriptors, using_unet_pipe=True):

    game, num_tiles, tileset, height, width, path_to_json = resolve_game(args)
    
    # MM2 checkpoints need the MM2 tools, not the SMB defaults (as in main()).
    assign_caption_fn = compare_captions_fn = None
    if "mm2" in os.path.basename(tileset).lower():
        assign_caption_fn, compare_captions_fn = mm2_caption_tools(tileset)

    width_range = resolve_eval_width_range(args)

    checkpoint_dirs = [
        (int(d.split("-")[-1]), os.path.join(args.model_path, d))
        for d in os.listdir(args.model_path)
        if os.path.isdir(os.path.join(args.model_path, d)) and d.startswith("checkpoint-")
    ]
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: x[0])
    if os.path.isdir(os.path.join(args.model_path, "unet")):
        checkpoint_dirs.append((checkpoint_dirs[-1][0] + 1, args.model_path))

    # Prepare output paths
    scores_jsonl_path = os.path.join(args.model_path, f"{os.path.basename(path_to_json).split('.')[0]}_scores_by_epoch.jsonl")
    plot_png_path = os.path.join(args.model_path, f"{os.path.basename(path_to_json).split('.')[0]}_caption_scores_plot.png")
    # Companion plot: one caption-adherence line per scene width, so weaknesses at a particular
    # size are visible. Only meaningful when the eval set spans multiple widths.
    width_plot_png_path = os.path.join(args.model_path, f"{os.path.basename(path_to_json).split('.')[0]}_caption_scores_by_width_plot.png")

    # Handle file existence based on resume flag
    completed_epochs = set()
    if os.path.exists(scores_jsonl_path):
        if args.resume:
            # Create backup files with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            if os.path.exists(scores_jsonl_path):
                backup_jsonl = scores_jsonl_path.replace('.jsonl', f'_backup_{timestamp}.jsonl')
                os.rename(scores_jsonl_path, backup_jsonl)
                # Copy content back to original file
                with open(backup_jsonl, 'r') as src, open(scores_jsonl_path, 'w') as dst:
                    for line in src:
                        entry = json.loads(line)
                        completed_epochs.add(entry['epoch'])
                        dst.write(line)
            if os.path.exists(plot_png_path):
                backup_png = plot_png_path.replace('.png', f'_backup_{timestamp}.png')
                os.rename(plot_png_path, backup_png)
        else:
            print(f"Error: Output files already exist. Use --resume to continue from previous run.")
            exit(1)

    # Initialize Plotter
    plotter = Plotter(
        log_file=scores_jsonl_path,
        update_interval=0.1,
        left_key="score",
        right_key=None,
        left_label="Caption Score",
        right_label=None,
        output_png=plot_png_path
    )

    # Start plotting in a background thread
    import threading
    plot_thread = threading.Thread(target=plotter.start_plotting)
    plot_thread.daemon = True
    plotter.running = True
    plot_thread.start()
    scores_by_epoch = []
    with open(scores_jsonl_path, "a") as f:
        for epoch, checkpoint_dir in tqdm(checkpoint_dirs, desc="Evaluating Checkpoints"):
            if epoch in completed_epochs:
                print(f"Skipping already evaluated checkpoint: {checkpoint_dir}")
                continue
                
            print(f"Evaluating checkpoint: {checkpoint_dir}")
            
            pipe = get_pipeline(checkpoint_dir).to(device)

            per_width_scores = {}
            avg_score, _, _, _ = calculate_caption_score_and_samples(
                device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, args.describe_absence, output=False, width=width, height=height, random_width=args.random_width, width_range=width_range, match_scene_width=args.match_scene_width, per_width_scores=per_width_scores, game=game,
                assign_caption_fn=assign_caption_fn, compare_captions_fn=compare_captions_fn # Jacob: Still not certain we need these two parameters
            )

            # Collapse the per-width score lists into mean scores for this checkpoint.
            width_scores = {w: sum(s) / len(s) for w, s in per_width_scores.items() if s}

            print(f"Checkpoint {checkpoint_dir} - Average caption adherence score: {avg_score:.4f}")
            if len(width_scores) > 1:
                print("  By scene width: " + ", ".join(f"{w}:{width_scores[w]:.4f}" for w in sorted(width_scores)))
            result = {"epoch": epoch, "score": avg_score, "checkpoint_dir": checkpoint_dir, "width_scores": width_scores}
            f.write(json.dumps(result) + "\n")
            f.flush()  # Ensure it's written immediately

            scores_by_epoch.append((epoch, avg_score, checkpoint_dir))

            # Update the plots after each checkpoint
            plotter.update_plot()
            plot_scores_by_width(scores_jsonl_path, width_plot_png_path)

    plotter.stop_plotting()
    plot_thread.join(timeout=1)

    # Final redraw (covers the resume case where every checkpoint was already evaluated).
    plot_scores_by_width(scores_jsonl_path, width_plot_png_path)

    return scores_by_epoch

def calculate_caption_score_and_samples(device, pipe, dataloader, inference_steps, guidance_scale, random_seed, id_to_char, char_to_id, tile_descriptors, describe_absence, height, width, output=True, random_width=False, width_range=None, match_scene_width=False, per_width_scores=None, compute_score=True, game=None, assign_caption_fn=None, compare_captions_fn=None):
    # compute_score=False skips deriving a structured caption from each generated scene and scoring
    # it against the prompt. Use it for natural-language (LLM) captions, where that comparison is
    # meaningless. Samples and prompts are still collected; avg_score is returned as None.

    # Optional per-width score collection. When the caller passes a dict, each sample's caption
    # score is appended under the width it was generated at (per_width_scores[width] -> list of
    # scores). This lets callers break the overall adherence score down by scene size without
    # changing this function's return signature (which several training scripts depend on).

    # Jacob: Not sure we need these: Can't we just use the game parameter instead?
    # assign_caption_fn/compare_captions_fn override the game detection below, for
    # Mario Maker (which the height checks can't tell from Mario: both MARIO_HEIGHT).

    #Used for potential level scene pruning later
    original_mode = dataloader.dataset.mode

    # --match_scene_width reads the per-batch width from the source scene (the batch is bucketed
    # to one width). Only meaningful when scenes are present (diff_text mode).
    match_scene_width = match_scene_width and original_mode == "diff_text"

    # When random_width is on, draw one width per batch (keeping each batch uniform) from
    # width_range, snapped to a size the UNet can denoise. A dedicated seeded RNG keeps the
    # per-batch width sequence identical across checkpoints so comparisons stay fair.
    if random_width and not isinstance(pipe, FDMPipeline):
        if width_range is None:
            raise ValueError("random_width=True requires width_range=(min_width, max_width)")
        width_factor = unet_width_factor(pipe.unet)
        width_rng = random.Random(random_seed)
    else:
        random_width = False

    score_sum = 0.0
    total_count = 0
    all_samples = []
    all_prompts = []
    compare_all_scores = []
    for batch_idx, batch in enumerate(dataloader):

        # Capture the source scene width before the scene is pruned out of the batch below.
        # The batch is bucketed to one width, so the first scene's width covers the whole batch.
        source_width = batch[0].shape[-1] if match_scene_width else None

        #Prune the one hot encoded level scene out of the batch if diff_text is being used
        if original_mode == "diff_text":
            batch = batch[1:]
            if len(batch)==1:
                batch=batch[0]

        # One width per batch so every sample in the batch shares a shape (required for batching).
        if match_scene_width:
            batch_width = source_width
        elif random_width:
            batch_width = sample_random_width(width_range[0], width_range[1], width_factor, width_rng)
        else:
            batch_width = width

        with torch.no_grad():  # Disable gradient computation to save memory
            if dataloader.dataset.negative_captions:
                # For negative captions, batch is (positive_captions, negative_captions)
                positive_captions, negative_captions = batch  # Unpack the batch directly
                param_values = {
                    "caption": list(positive_captions),
                    "negative_prompt": list(negative_captions),
                    "num_inference_steps": inference_steps,
                    "height": height,
                    "width": batch_width,
                    "guidance_scale": guidance_scale,
                    "output_type": "tensor",
                    "batch_size": len(positive_captions)
                }
            elif isinstance(pipe, FDMPipeline):
                param_values = {
                    "caption": list(batch),
                    "batch_size": len(batch),
                }
            else:
                param_values = {
                    "caption": list(batch),
                    "num_inference_steps": inference_steps,
                    "height": height,
                    "width": batch_width,
                    "guidance_scale": guidance_scale,
                    "output_type": "tensor",
                    "batch_size": len(batch)
                }

            generator = torch.Generator(device).manual_seed(int(random_seed))
            # Generate a batch of samples at once
            samples = pipe(generator=generator, **param_values).images  # (batch_size, ...)
            #print("samples.shape", samples.shape)
            for i in range(len(samples)):
                if dataloader.dataset.negative_captions:
                    caption = positive_captions[i]
                else:
                    caption = batch[i]
                    
                all_prompts.append(caption)

                # For LLM/natural-language prompts there is no structured caption to derive or
                # compare, so just keep the generated sample and move on.
                if not compute_score:
                    all_samples.append(samples[i])
                    total_count += 1
                    continue

                sample = samples[i].unsqueeze(0)
                #print("sample.shape", sample.shape)
                sample_indices = convert_to_level_format(sample)
                #print("first sample_indices", sample_indices[0])
                scene = sample_indices[0].tolist()  # Always just one scene: (1,16,16)
                #quit()

                if assign_caption_fn is not None:
                    actual_caption = assign_caption_fn(scene)
                elif game == "LR":
                    scene = [[tile % common_settings.LR_TILE_COUNT for tile in s] for s in scene]
                    actual_caption = lr_assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, describe_absence)
                elif game in ("MM-Simple", "MM-Full", "MMLV"):
                    actual_caption = mm_assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, describe_absence)
                elif game in ("MM2", "Mario"): # Jacob: This is new: I think Mario and Mario Maker can call the same function.
                    actual_caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, describe_absence)
                else: # Jacob: Added this failure case
                    raise ValueError(f"Unknown game type: {game}")


                if output: print(f"\t{caption}")
                if compare_captions_fn is not None:
                    compare_score = compare_captions_fn(caption, actual_caption)
                elif game == "LR":
                    compare_score = lr_compare_captions(caption, actual_caption)
                elif game in ("MM-Simple", "MM-Full", "MMLV"):
                    compare_score = mm_compare_captions(caption, actual_caption)
                elif game in ("MM2", "Mario"): # Jacob: This is new: I think Mario and Mario Maker can call the same function.
                    compare_score = compare_captions(caption, actual_caption)
                else: # Jacob: Added this failure case
                    raise ValueError(f"Unknown game type: {game}")

                if output: print(f"\tcompare_score: {compare_score}")
                compare_all_scores.append(compare_score)

                # Record this sample's score against the width it was generated at, so callers
                # can plot/inspect adherence separately for each scene size.
                if per_width_scores is not None:
                    per_width_scores.setdefault(batch_width, []).append(compare_score)

                score_sum += compare_score
                total_count += 1

                all_samples.append(samples[i])  # (channels, height, width); stacked/kept-as-list below
                del sample, sample_indices, scene, actual_caption  # Remove unused variables

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU VRAM cache

        if output: print(f"Batch {batch_idx+1}/{len(dataloader)}:")

    avg_score = (score_sum / total_count) if compute_score and total_count else None
    # Stack all per-sample (C,H,W) tensors into one (N,C,H,W) batch. With random_width the
    # widths differ across batches and can't be stacked, so keep a list of (C,H,W) tensors;
    # downstream samples_to_scenes / per-sample visualization handle either form.
    if len({tuple(s.shape) for s in all_samples}) == 1:
        all_samples = torch.stack(all_samples, dim=0)[:total_count]
    else:
        all_samples = all_samples[:total_count]

    dataloader.dataset.mode=original_mode

    return (avg_score, all_samples, all_prompts, compare_all_scores) 
    # Adding this return value broke code in MANY places. Cannot do this unless you make sure that all calls to this function expect 4 values
    # Found all references of this method and made them all return 4 values

if __name__ == "__main__":
    main()
