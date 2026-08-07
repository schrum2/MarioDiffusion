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


def load_clip_model(model_name, device):
    """Load a pretrained CLIP model/processor from Hugging Face for computing CLIPScore-style
    text-image alignment. Imported lazily so `transformers` is only required when
    --use_clip_score is actually passed."""
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    return clip_model, clip_processor


def compute_clip_score(image, text, clip_model, clip_processor, device):
    """Cosine similarity between the CLIP image embedding of a rendered scene and the CLIP
    text embedding of the caption used to generate it. Returns a plain Python float
    (roughly in [-1, 1], typically small and positive for related image/text pairs)."""
    image_embeds = compute_clip_image_embedding(image, clip_model, clip_processor, device)
    text_embeds = compute_clip_text_embedding(text, clip_model, clip_processor, device)
    similarity = (image_embeds * text_embeds).sum(dim=-1)
    return similarity.item()


def compute_clip_image_embedding(image, clip_model, clip_processor, device):
    """Return the normalized CLIP image embedding for a single PIL image."""
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_embeds = clip_model.get_image_features(**inputs)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds


def compute_clip_text_embedding(text, clip_model, clip_processor, device):
    """Return the normalized CLIP text embedding for a single caption string."""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_embeds = clip_model.get_text_features(**inputs)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    return text_embeds


def compute_clip_image_similarity(image_a, image_b, clip_model, clip_processor, device):
    """Cosine similarity between two CLIP image embeddings."""
    image_a_embeds = compute_clip_image_embedding(image_a, clip_model, clip_processor, device)
    image_b_embeds = compute_clip_image_embedding(image_b, clip_model, clip_processor, device)
    similarity = (image_a_embeds * image_b_embeds).sum(dim=-1)
    return similarity.item()


def render_scene_image(sample, game):
    """Render a single generated sample (channels, height, width) to a PIL image via the
    existing tile-based visualizer, for use with CLIP scoring. Mirrors the same
    visualize_samples call pattern used elsewhere in this file for saving per-sample PNGs,
    just without writing anything to disk (output_dir=None returns the PIL image directly)."""
    return visualize_samples(sample.unsqueeze(0), output_dir=None, game=game)


def render_scene_image_from_scene(scene, game):
    """Render a source-level scene grid (list of rows or a numpy array) to a PIL image."""
    if scene is None:
        return None
    scene_array = np.asarray(scene, dtype=np.int64)
    if scene_array.ndim != 2:
        raise ValueError(f"Expected a 2D scene grid, got shape {scene_array.shape}")
    if scene_array.size == 0:
        return None

    num_tiles = int(scene_array.max()) + 1 if scene_array.size else 1
    scene_tensor = torch.zeros((num_tiles, scene_array.shape[0], scene_array.shape[1]), dtype=torch.float32)
    for row_idx, row in enumerate(scene_array):
        for col_idx, tile_id in enumerate(row):
            tile_id = int(tile_id)
            if tile_id < 0 or tile_id >= num_tiles:
                raise ValueError(f"Tile ID {tile_id} is out of range for a {num_tiles}-tile scene")
            scene_tensor[tile_id, row_idx, col_idx] = 1.0
    return visualize_samples(scene_tensor.unsqueeze(0), output_dir=None, game=game)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate caption adherence for a pretrained text-conditional diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--game", type=str, default=None, choices=["Mario", "LR", "MM-Simple", "MM-Full", "MMLV", "MM2"], help="Game to evaluate: selects the tileset, scene shape, tile count, and the tiles used for rendering. This is the main way to pick a game, and how Mega Man should be resolved. When omitted, the game is derived from --num_tiles (+ --mm) for backward compatibility.")
    parser.add_argument("--num_tiles", type=int, default=common_settings.MARIO_TILE_COUNT, help="Number of tile types")
    # Note: "--mm"/MM-Simple/MM-Full are Mega Man; the "MM2" game is Mario Maker 2.
    parser.add_argument("--mm", action="store_true", help="Backward-compatible shorthand for Mega Man when --game is not given: routes the 13-tile case to MM-Simple instead of Mario (they share a tile count). Prefer --game MM-Simple / --game MM-Full.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
        
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--all_captions", action="store_true", help="Generate a scene for EVERY stored caption of each entry (caption, caption1, caption2, ...), not just the first. Output entries carry source_index/source_name/caption_index so scenes generated from the same source scene can be grouped. Mario Maker only")
    parser.add_argument("--caption_source_keys", nargs="+", type=str, default=None, help="Generate one scene per caption stored under the listed source keys in the dataset JSON. Each output entry is tagged with the source entry, caption key, and caption index so prompt/scene pairs from the same input scene can be grouped.")
    parser.add_argument("--inference_steps", type=int, default=common_settings.NUM_INFERENCE_STEPS, help="Number of denoising steps") # Large reduction from the 500 used during training
    parser.add_argument("--guidance_scale", type=float, default=common_settings.GUIDANCE_SCALE, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--save_as_json", action="store_true", help="Save generated levels as JSON")
    parser.add_argument("--no_caption_score", action="store_true", help="Skip the caption-adherence score calculation and just generate a sample for each prompt.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted checkpoint comparison run")

    # CLIP-based alignment scoring (off by default). Complements the deterministic
    # caption-adherence score above and, unlike it, works for free-form LLM captions since it
    # doesn't rely on parsing the caption back into structured claims.
    parser.add_argument("--no_clip_score", action="store_false", dest="use_clip_score", help="Disable CLIP scoring (enabled by default).")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32", help="Hugging Face CLIP model to use when --use_clip_score is set.")

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

class PromptListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
        self.mode = "text"
        self.negative_captions = False
        self.data = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]["prompt"]


def resolve_game(args):
    """
    Map the CLI args to (game, num_tiles, tileset, height, width, path_to_json).
    """
    config = common_settings.get_game_config(args.game)
    
    num_tiles = config["tile_count"] 
    tileset = config["tileset"]
    height = config["height"]
    width = config["width"]
    
    return num_tiles, tileset, height, width

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
    game = args.game
    num_tiles, tileset, height, width = resolve_game(args)

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

    # Load the CLIP model once and reuse it across every generation pathway below.
    clip_model, clip_processor = (None, None)
    if args.use_clip_score:
        clip_model, clip_processor = load_clip_model(args.clip_model_name, device)

    if args.match_scene_width and args.random_width:
        print("Error: --match_scene_width and --random_width are mutually exclusive.")
        exit(1)

    # Mario Maker levels get their own captioner and comparison (the SMB code
    # would call every MM2 pipe broken -- the tileset has no <>[] chars).
    mm2 = "mm2" in os.path.basename(tileset).lower()

    # At least some captions will be from an LLM, and caption adherence can't be computed for those.
    if args.all_captions and args.compare_checkpoints:
        print("Error: --all_captions cannot be combined with --compare_checkpoints.")
        exit(1)

    # TODO: this MM2-specific path could be generalized by carrying metadata in LevelDataset.
    if mm2 and not args.compare_checkpoints:
        # Generate straight from the dataset entries (not a LevelDataset) so each
        # output can be tagged with the source scene its caption came from.
        with open(args.json, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        items = expand_mm2_caption_items(raw_data, args.all_captions)
        if not items:
            print(f"Error: no captions found in {args.json}")
            exit(1)
        print(f"Generating {len(items)} scenes from {len(raw_data)} dataset entries...")
        avg_score, avg_clip_score, avg_scene_clip_score, results = mm2_caption_adherence(
            args, device, pipe, items, tileset,
            compute_clip=args.use_clip_score, clip_model=clip_model, clip_processor=clip_processor
        )
        if avg_score is not None:
            print(f"Average caption adherence score: {avg_score:.4f}")
        if avg_clip_score is not None:
            print(f"Average CLIP score: {avg_clip_score:.4f}")
        if avg_scene_clip_score is not None:
            print(f"Average scene CLIP score: {avg_scene_clip_score:.4f}")
        if args.save_as_json:
            out_path = os.path.join(args.output_dir, "all_levels.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved {len(results)} captioned scenes to {out_path}")
        return

    # Load once. LevelDataset.data holds the raw entries (scenes included) regardless of mode,
    # so we can inspect the set of scene widths here to decide how to generate.
    if args.caption_source_keys:
        with open(args.json, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        prompt_items = expand_caption_items(raw_data, args.caption_source_keys)
        if not prompt_items:
            print(f"Error: no captions found under the requested caption_source_keys in {args.json}")
            exit(1)
        prompt_dataset = PromptListDataset(prompt_items)
        dataloader = DataLoader(
            prompt_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
        prompt_metadata = [item["metadata"] for item in prompt_items]
        scene_widths = {
            len(item["scene"][0])
            for item in raw_data
            if isinstance(item, dict) and item.get("scene") is not None
        }
    else:
        dataset = LevelDataset(
            json_path=args.json,
            tokenizer=None,
            shuffle=False,
            mode="text",
            augment=False,
            # num_tiles comes from resolve_game so it always matches the selected game's tileset.
            num_tiles=num_tiles,
        )
        prompt_metadata = None
        scene_widths = {len(item["scene"][0]) for item in dataset.data if isinstance(item, dict) and item.get("scene") is not None}

        if args.match_scene_width and not scene_widths:
            print(f"Error: --match_scene_width requires a scene-bearing dataset, but '{args.json}' has caption-only entries.")
            exit(1)

        # Datasets with more than one scene shape default to recreating each caption at its source
        # scene's width. Homogeneous datasets (one width) and caption-only sets are left on the old
        # fixed-width path. --match_scene_width forces it on; --random_width opts out.
        if len(scene_widths) > 1 and not args.random_width and not args.match_scene_width:
            print(f"Detected {len(scene_widths)} scene widths {sorted(scene_widths)} in {os.path.basename(args.json)}; matching generation width to each source scene.")
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
        scores_by_epoch = track_caption_adherence(args, device, dataloader, id_to_char, char_to_id, tile_descriptors, clip_model=clip_model, clip_processor=clip_processor)

    else:
        # Just run on one model and get samples as well
        width_range = resolve_eval_width_range(args)
        per_width_scores = {}
        result = calculate_caption_score_and_samples(device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, args.describe_absence, output=False, height=height, width=width, random_width=args.random_width, width_range=width_range, match_scene_width=args.match_scene_width, per_width_scores=per_width_scores, compute_score=not args.no_caption_score, game=game, prompt_metadata=prompt_metadata, compute_clip=args.use_clip_score, clip_model=clip_model, clip_processor=clip_processor)
        avg_score = result["avg_score"]
        all_samples = result["all_samples"]
        all_prompts = result["all_prompts"]
        compare_all_scores = result["compare_all_scores"]
        clip_all_scores = result["clip_all_scores"]
        scene_clip_all_scores = result["scene_clip_all_scores"]

        if avg_score is not None:
            print(f"Average caption adherence score: {avg_score:.4f}")
        if clip_all_scores:
            avg_clip_score = sum(clip_all_scores) / len(clip_all_scores)
            print(f"Average CLIP score: {avg_clip_score:.4f}")
        if scene_clip_all_scores:
            scene_clip_values = [score for score in scene_clip_all_scores if score is not None]
            if scene_clip_values:
                avg_scene_clip_score = sum(scene_clip_values) / len(scene_clip_values)
                print(f"Average scene CLIP score: {avg_scene_clip_score:.4f}")
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
            # compare_all_scores is only populated per-sample when caption-adherence scoring
            # ran (i.e. not args.no_caption_score); otherwise leave "score" out of each entry.
            have_compare_scores = bool(compare_all_scores) and len(compare_all_scores) == len(scenes)
            have_clip_scores = bool(clip_all_scores) and len(clip_all_scores) == len(scenes)
            have_scene_clip_scores = bool(scene_clip_all_scores) and len(scene_clip_all_scores) == len(scenes)
            paired = []
            for idx, (prompt, scene, metadata) in enumerate(zip(all_prompts, scenes, prompt_metadata or [None] * len(scenes))):
                if game == "LR":
                    capt_scene = [[tile % common_settings.LR_TILE_COUNT for tile in row] for row in scene]
                    caption = lr_assign_caption(capt_scene, id_to_char, char_to_id, tile_descriptors, False, args.describe_absence)
                elif game in ("MM-Simple", "MM-Full", "MMLV"):
                    caption = mm_assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, args.describe_absence)
                else:  # Mario
                    caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, args.describe_absence)
                entry = {"prompt": prompt, "caption": caption, "scene": scene}
                if have_compare_scores:
                    entry["score"] = compare_all_scores[idx]
                if have_clip_scores:
                    entry["clip_score"] = clip_all_scores[idx]
                if have_scene_clip_scores:
                    entry["scene_clip_score"] = scene_clip_all_scores[idx]
                if metadata:
                    entry.update(metadata)
                paired.append(entry)
            with open(os.path.join(args.output_dir, "all_levels.json"), "w") as f:
                json.dump(paired, f, indent=4)


def expand_caption_items(data, caption_source_keys=None):
    """Create one generation item per caption pulled from the requested source keys."""
    items = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue

        source_name = entry.get("name") or entry.get("id") or f"entry_{idx}"
        source_group_id = f"{idx}:{source_name}"
        if caption_source_keys:
            for key in caption_source_keys:
                value = entry.get(key)
                if isinstance(value, list):
                    captions = [c for c in value if c]
                elif isinstance(value, str) and value:
                    captions = [value]
                else:
                    continue

                for cidx, caption in enumerate(captions):
                    items.append({
                        "prompt": caption,
                        "scene": entry.get("scene"),
                        "metadata": {
                            "caption": caption,
                            "caption_source_key": key,
                            "caption_index": cidx,
                            "source_index": idx,
                            "source_id": source_name,
                            "source_name": source_name,
                            "source_group_id": source_group_id,
                        },
                    })
        elif entry.get("caption"):
            items.append({
                "prompt": entry["caption"],
                "scene": entry.get("scene"),
                "metadata": {
                    "caption": entry["caption"],
                    "caption_source_key": "caption",
                    "caption_index": 0,
                    "source_index": idx,
                    "source_id": source_name,
                    "source_name": source_name,
                    "source_group_id": source_group_id,
                },
            })
    return items


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


def mm2_caption_adherence(args, device, pipe, items, tileset, compute_clip=False, clip_model=None, clip_processor=None):
    """Generate one scene per item and score it with the MM2 captioner. Batches
    are bucketed by scene size (an item's source scene shape, else
    args.height x args.width) since a batch must share one shape. Returns
    (average score, average clip score, average scene-CLIP score, results); each result carries
    the prompt, scene, its deterministic caption, the score, the clip score (if computed),
    the scene-based CLIP score (when available), and the source metadata. Caption-adherence
    scoring (the deterministic recaptioning + compare) is skipped when args.no_caption_score
    is set, matching the other generation pathways; avg_score is then None and results omit
    "caption"/"score"."""
    compute_score = not args.no_caption_score
    assign_caption_fn = compare_captions_fn = None
    if compute_score:
        assign_caption_fn, compare_captions_fn = mm2_caption_tools(tileset)

    by_shape = {}
    for item in items:
        if item.get("scene") is not None:
            shape = (len(item["scene"]), len(item["scene"][0]))
        else:
            shape = (args.height, args.width)
        by_shape.setdefault(shape, []).append(item)

    results = []
    score_sum = 0.0
    clip_score_sum = 0.0
    clip_count = 0
    scene_clip_score_sum = 0.0
    scene_clip_count = 0
    per_shape_scores = {}
    scene_embedding_cache = {}
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
            for idx_in_batch, (item, scene) in enumerate(zip(batch, scenes)):
                entry = {
                    "prompt": item["caption"],
                    "scene": scene,
                    "source_index": item["source_index"],
                    "source_name": item["source_name"],
                    "caption_index": item["caption_index"],
                }
                if compute_score:
                    actual_caption = assign_caption_fn(scene)
                    score = compare_captions_fn(item["caption"], actual_caption)
                    score_sum += score
                    per_shape_scores.setdefault(shape, []).append(score)
                    entry["caption"] = actual_caption
                    entry["score"] = score
                if compute_clip:
                    sample_image = render_scene_image(samples[idx_in_batch], "MM2")
                    sample_image_embed = compute_clip_image_embedding(sample_image, clip_model, clip_processor, device)
                    text_embed = compute_clip_text_embedding(item["caption"], clip_model, clip_processor, device)
                    clip_score = (sample_image_embed * text_embed).sum(dim=-1).item()
                    entry["clip_score"] = clip_score
                    clip_score_sum += clip_score
                    clip_count += 1

                    scene_clip_score = None
                    source_scene = item.get("scene")
                    if source_scene is not None:
                        scene_key = tuple(tuple(int(tile) for tile in row) for row in source_scene)
                        if scene_key not in scene_embedding_cache:
                            scene_reference_image = render_scene_image_from_scene(source_scene, "MM2")
                            scene_embedding_cache[scene_key] = compute_clip_image_embedding(scene_reference_image, clip_model, clip_processor, device)
                        scene_clip_score = (sample_image_embed * scene_embedding_cache[scene_key]).sum(dim=-1).item()
                        scene_clip_score_sum += scene_clip_score
                        scene_clip_count += 1
                    entry["scene_clip_score"] = scene_clip_score
                results.append(entry)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(per_shape_scores) > 1:
        print("Samples and caption adherence by scene size:")
        for shape in sorted(per_shape_scores):
            scores = per_shape_scores[shape]
            print(f"\t{shape[0]}x{shape[1]}: {len(scores)} samples, avg score {sum(scores) / len(scores):.4f}")

    avg_score = (score_sum / len(results)) if compute_score and results else None
    avg_clip_score = (clip_score_sum / clip_count) if compute_clip and clip_count else None
    avg_scene_clip_score = (scene_clip_score_sum / scene_clip_count) if compute_clip and scene_clip_count else None
    return avg_score, avg_clip_score, avg_scene_clip_score, results


def track_caption_adherence(args, device, dataloader, id_to_char, char_to_id, tile_descriptors, using_unet_pipe=True, clip_model=None, clip_processor=None):

    _, tileset, height, width = resolve_game(args)
    
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
    scores_jsonl_path = os.path.join(args.model_path, f"{os.path.basename(args.json).split('.')[0]}_scores_by_epoch.jsonl")
    plot_png_path = os.path.join(args.model_path, f"{os.path.basename(args.json).split('.')[0]}_caption_scores_plot.png")
    # Companion plot: one caption-adherence line per scene width, so weaknesses at a particular
    # size are visible. Only meaningful when the eval set spans multiple widths.
    width_plot_png_path = os.path.join(args.model_path, f"{os.path.basename(args.json).split('.')[0]}_caption_scores_by_width_plot.png")
    # CLIP-score companion plot (only produced when --use_clip_score is set). Derived from the
    # same scores_jsonl_path file as the caption-adherence plot above, just a different key.
    clip_plot_png_path = os.path.join(args.model_path, f"{os.path.basename(args.json).split('.')[0]}_clip_scores_plot.png")

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

    # Companion Plotter for CLIP score, reading the same jsonl file but a different key.
    # Only created when --use_clip_score is set, since the jsonl rows otherwise have no
    # "clip_score" field to plot.
    clip_plotter = None
    if args.use_clip_score:
        clip_plotter = Plotter(
            log_file=scores_jsonl_path,
            update_interval=0.1,
            left_key="clip_score",
            right_key=None,
            left_label="CLIP Score",
            right_label=None,
            output_png=clip_plot_png_path
        )

    # Start plotting in a background thread
    import threading
    plot_thread = threading.Thread(target=plotter.start_plotting)
    plot_thread.daemon = True
    plotter.running = True
    plot_thread.start()

    clip_plot_thread = None
    if clip_plotter is not None:
        clip_plot_thread = threading.Thread(target=clip_plotter.start_plotting)
        clip_plot_thread.daemon = True
        clip_plotter.running = True
        clip_plot_thread.start()

    scores_by_epoch = []
    with open(scores_jsonl_path, "a") as f:
        for epoch, checkpoint_dir in tqdm(checkpoint_dirs, desc="Evaluating Checkpoints"):
            if epoch in completed_epochs:
                print(f"Skipping already evaluated checkpoint: {checkpoint_dir}")
                continue
                
            print(f"Evaluating checkpoint: {checkpoint_dir}")
            
            pipe = get_pipeline(checkpoint_dir).to(device)

            per_width_scores = {}
            clip_all_scores = [] if args.use_clip_score else None
            scene_clip_all_scores = [] if args.use_clip_score else None
            # Pass the MM2 caption tools (None for other games) so MM2 scores with the MM2 captioner;
            # scene shape alone can't tell MM2 from Mario.
            avg_score = calculate_caption_score_and_samples(
                device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, 
                args.describe_absence, output=False, width=width, height=height, random_width=args.random_width, 
                width_range=width_range, match_scene_width=args.match_scene_width, per_width_scores=per_width_scores, game=args.game,
                assign_caption_fn=assign_caption_fn, compare_captions_fn=compare_captions_fn,
                compute_clip=args.use_clip_score, clip_model=clip_model, clip_processor=clip_processor
            )["avg_score"]

            # Collapse the per-width score lists into mean scores for this checkpoint.
            width_scores = {w: sum(s) / len(s) for w, s in per_width_scores.items() if s}

            avg_clip_score = (sum(clip_all_scores) / len(clip_all_scores)) if clip_all_scores else None
            scene_clip_values = [score for score in scene_clip_all_scores if score is not None] if scene_clip_all_scores is not None else []
            avg_scene_clip_score = (sum(scene_clip_values) / len(scene_clip_values)) if scene_clip_values else None

            print(f"Checkpoint {checkpoint_dir} - Average caption adherence score: {avg_score:.4f}")
            if avg_clip_score is not None:
                print(f"Checkpoint {checkpoint_dir} - Average CLIP score: {avg_clip_score:.4f}")
            if avg_scene_clip_score is not None:
                print(f"Checkpoint {checkpoint_dir} - Average scene CLIP score: {avg_scene_clip_score:.4f}")
            if len(width_scores) > 1:
                print("  By scene width: " + ", ".join(f"{w}:{width_scores[w]:.4f}" for w in sorted(width_scores)))
            result = {"epoch": epoch, "score": avg_score, "checkpoint_dir": checkpoint_dir, "width_scores": width_scores}
            if avg_clip_score is not None:
                result["clip_score"] = avg_clip_score
            if avg_scene_clip_score is not None:
                result["scene_clip_score"] = avg_scene_clip_score
            f.write(json.dumps(result) + "\n")
            f.flush()  # Ensure it's written immediately

            scores_by_epoch.append((epoch, avg_score, checkpoint_dir))

            # Update the plots after each checkpoint
            plotter.update_plot()
            plot_scores_by_width(scores_jsonl_path, width_plot_png_path)
            if clip_plotter is not None:
                clip_plotter.update_plot()

    plotter.stop_plotting()
    plot_thread.join(timeout=1)
    if clip_plotter is not None:
        clip_plotter.stop_plotting()
        clip_plot_thread.join(timeout=1)

    # Final redraw (covers the resume case where every checkpoint was already evaluated).
    plot_scores_by_width(scores_jsonl_path, width_plot_png_path)

    return scores_by_epoch

def calculate_caption_score_and_samples(device, pipe, dataloader, inference_steps, guidance_scale, random_seed, id_to_char, char_to_id, tile_descriptors, describe_absence, height, width, output=True, random_width=False, width_range=None, match_scene_width=False, per_width_scores=None, compute_score=True, game=None, assign_caption_fn=None, compare_captions_fn=None, prompt_metadata=None, compute_clip=False, clip_model=None, clip_processor=None):
    # compute_clip=True additionally renders each generated sample to an image (via the same
    # tile-based visualizer used for saving PNGs) and scores it against the prompt used to
    # generate it with CLIPScore-style cosine similarity. This is independent of compute_score:
    # it works for both structured and free-form (LLM) captions, since it doesn't rely on
    # deriving a structured caption from the generated scene. Per-sample scores are appended,
    # in generation order, to clip_all_scores list.
    # compute_score=False skips deriving a structured caption from each generated scene and scoring
    # it against the prompt. Use it for natural-language (LLM) captions, where that comparison is
    # meaningless. Samples and prompts are still collected; avg_score is returned as None.

    # Optional per-width score collection. When the caller passes a dict, each sample's caption
    # score is appended under the width it was generated at (per_width_scores[width] -> list of
    # scores). This lets callers break the overall adherence score down by scene size without
    # changing this function's return signature (which several training scripts depend on).

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
    clip_all_scores = []         
    scene_clip_all_scores = []  
    prompt_index = 0
    scene_embedding_cache = {}
    for batch_idx, batch in enumerate(dataloader):

        # The raw collated `batch` from the DataLoader doesn't reliably reflect the number of
        # samples: in "diff_text" mode it's a 2-tuple (scenes, captions), not a per-sample list.
        if original_mode == "diff_text":
            actual_batch_size = len(batch[1]) if len(batch) > 1 else len(batch[0])
        else:
            actual_batch_size = len(batch)

        batch_metadata = None
        batch_source_scenes = None
        if prompt_metadata is not None:
            batch_metadata = prompt_metadata[prompt_index:prompt_index + actual_batch_size]
        if compute_clip:
            batch_source_scenes = (
                dataloader.dataset.data[prompt_index:prompt_index + actual_batch_size]
                if hasattr(dataloader.dataset, "data") else None
            )
        prompt_index += actual_batch_size

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

                # CLIP scoring is independent of compute_score (works for LLM captions too),
                # so it runs here before the compute_score early-continue below.
                if compute_clip:
                    sample_image = render_scene_image(samples[i], game)
                    sample_image_embed = compute_clip_image_embedding(sample_image, clip_model, clip_processor, device)
                    text_embed = compute_clip_text_embedding(caption, clip_model, clip_processor, device)
                    clip_score = (sample_image_embed * text_embed).sum(dim=-1).item()
                    clip_all_scores.append(clip_score)

                    scene_clip_score = None
                    if batch_source_scenes is not None:
                        source_scene = batch_source_scenes[i].get("scene") if isinstance(batch_source_scenes[i], dict) else None
                        if source_scene is not None:
                            scene_key = tuple(tuple(int(tile) for tile in row) for row in source_scene)
                            if scene_key not in scene_embedding_cache:
                                scene_reference_image = render_scene_image_from_scene(source_scene, game)
                                scene_embedding_cache[scene_key] = compute_clip_image_embedding(scene_reference_image, clip_model, clip_processor, device)
                            scene_clip_score = (sample_image_embed * scene_embedding_cache[scene_key]).sum(dim=-1).item()

                    scene_clip_all_scores.append(scene_clip_score)

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

                # MM2 uses assign_caption_fn (set above), so it never reaches these by-game cases.
                if assign_caption_fn is not None:
                    actual_caption = assign_caption_fn(scene)
                elif game == "LR":
                    scene = [[tile % common_settings.LR_TILE_COUNT for tile in s] for s in scene]
                    actual_caption = lr_assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, describe_absence)
                elif game in ("MM-Simple", "MM-Full", "MMLV"):
                    actual_caption = mm_assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, describe_absence)
                elif game == "Mario":
                    actual_caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, describe_absence)
                else:
                    raise ValueError(f"Unknown game type: {game}")


                if output: print(f"\t{caption}")
                # Same idea: MM2 uses compare_captions_fn above, so only the other games reach here.
                if compare_captions_fn is not None:
                    compare_score = compare_captions_fn(caption, actual_caption)
                elif game == "LR":
                    compare_score = lr_compare_captions(caption, actual_caption)
                elif game in ("MM-Simple", "MM-Full", "MMLV"):
                    compare_score = mm_compare_captions(caption, actual_caption)
                elif game == "Mario":
                    compare_score = compare_captions(caption, actual_caption)
                else:
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

    # CLIP averages are independent of compute_score/game.
    valid_clip_scores = clip_all_scores if clip_all_scores is not None else []
    avg_clip_score = (
        sum(valid_clip_scores) / len(valid_clip_scores)
        if len(valid_clip_scores) > 0 else None
    )

    valid_scene_clip_scores = (
        [s for s in scene_clip_all_scores if s is not None]
        if scene_clip_all_scores is not None else []
    )
    avg_scene_clip_score = (
        sum(valid_scene_clip_scores) / len(valid_scene_clip_scores)
        if len(valid_scene_clip_scores) > 0 else None
    )

    # Stack all per-sample (C,H,W) tensors into one (N,C,H,W) batch. With random_width the
    # widths differ across batches and can't be stacked, so keep a list of (C,H,W) tensors;
    # downstream samples_to_scenes / per-sample visualization handle either form.
    if len({tuple(s.shape) for s in all_samples}) == 1:
        all_samples = torch.stack(all_samples, dim=0)[:total_count]
    else:
        all_samples = all_samples[:total_count]

    dataloader.dataset.mode = original_mode

    result = dict()
    result["avg_score"] = avg_score
    result["avg_clip_score"] = avg_clip_score
    result["avg_scene_clip_score"] = avg_scene_clip_score
    result["all_samples"] = all_samples
    result["all_prompts"] = all_prompts
    result["compare_all_scores"] = compare_all_scores
    result["clip_all_scores"] = clip_all_scores if clip_all_scores is not None else None
    result["scene_clip_all_scores"] = scene_clip_all_scores if scene_clip_all_scores is not None else None

    return result

if __name__ == "__main__":
    main()