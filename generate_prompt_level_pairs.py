"""
Sample captions from a captions json (e.g. a *_RandomTest-*.json), generate a level for each one
with text_to_level_diffusion's generator, and save the prompt/level pairs as a json that
ascii_data_browser.py can open. Each entry holds:
    prompt  - the caption the level was generated from (shown with "Show Prompt" in the browser)
    caption - the caption assigned to the generated scene
    scene   - the generated scene
    seed, source_index - generation seed and the prompt's index in --captions_json

Example:
python generate_prompt_level_pairs.py --model_path Game_MMLV/10_20-conditional-mlm --captions_json Game_MMLV/DATA/10_20_RandomTest-regular.json --num_pairs 50 --save_images
python ascii_data_browser.py Game_MMLV/10_20-conditional-mlm/10_20_RandomTest-regular_pairs.json MMLV
"""
import argparse
import json
import os
import random
from types import SimpleNamespace

import torch

import util.common_settings as common_settings
from text_to_level_diffusion import InteractiveLevelGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prompt/level pairs from randomly sampled captions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained text-conditional diffusion model")
    parser.add_argument("--captions_json", type=str, required=True, help="Json list of entries with a 'caption' field (e.g. a RandomTest json)")
    parser.add_argument("--num_pairs", type=int, default=50, help="Number of captions to sample (capped at the number of captions available)")
    parser.add_argument("--game", type=str, default=None, choices=common_settings.GAME_CLI_CHOICES, help="Game the model was trained on. Defaults to the 'game' in the model's interactive_defaults.json")
    parser.add_argument("--seed", type=int, default=42, help="Seed for picking captions; pair i is generated with seed + i")
    parser.add_argument("--num_inference_steps", type=int, default=common_settings.NUM_INFERENCE_STEPS, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=common_settings.GUIDANCE_SCALE, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure in the assigned captions")
    parser.add_argument("--automatic_negative_captions", action="store_true", default=False, help="Automatically create negative captions for the prompts")
    parser.add_argument("--automatic_absence_captions", action="store_true", default=False, help="Automatically append absence captions to the prompts")
    parser.add_argument("--output_json", type=str, default=None, help="Where to write the pairs (default: <model_path>/<captions_json name>_pairs.json)")
    parser.add_argument("--save_images", action="store_true", default=False, help="Also save a png of each generated level in a folder next to --output_json")
    return parser.parse_args()


def resolve_game(args):
    if args.game:
        return args.game
    defaults_path = os.path.join(args.model_path, "interactive_defaults.json")
    if os.path.isfile(defaults_path):
        with open(defaults_path) as f:
            game = json.load(f).get("game")
        if game:
            return game
    raise ValueError(f"No --game given and no game found in {defaults_path}")


def main():
    args = parse_args()
    game = resolve_game(args)
    config = common_settings.get_game_config(game)

    with open(args.captions_json) as f:
        captions = [entry["caption"] for entry in json.load(f)]
    if args.num_pairs > len(captions):
        print(f"Only {len(captions)} captions in {args.captions_json}; generating {len(captions)} pairs")
    picked = random.Random(args.seed).sample(range(len(captions)), min(args.num_pairs, len(captions)))

    # Arguments InteractiveLevelGeneration reads off text_to_level_diffusion's parsed args
    gen_args = SimpleNamespace(
        model_path=args.model_path,
        game=game,
        tileset=config["tileset"],
        describe_absence=args.describe_absence,
        automatic_negative_captions=args.automatic_negative_captions,
        automatic_absence_captions=args.automatic_absence_captions,
        visualize=False,
    )
    generator_app = InteractiveLevelGeneration(gen_args, interactive=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_json = args.output_json or os.path.join(
        args.model_path, os.path.splitext(os.path.basename(args.captions_json))[0] + "_pairs.json")
    image_dir = os.path.splitext(output_json)[0] + "_images"
    if args.save_images:
        os.makedirs(image_dir, exist_ok=True)

    pairs = []
    for i, caption_index in enumerate(picked):
        seed = args.seed + i
        param_values = {
            "caption": captions[caption_index],
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
        }
        generator_app.get_extra_params(param_values)
        image = generator_app.generate_image(param_values, torch.Generator(device).manual_seed(seed))
        if image is None:
            print(f"Generation failed for caption {caption_index}; skipping")
            continue

        pairs.append({
            "prompt": captions[caption_index],
            "caption": generator_app.last_actual_caption,
            "scene": generator_app.last_scene,
            "seed": seed,
            "source_index": caption_index,
        })
        if args.save_images:
            image.save(os.path.join(image_dir, f"{len(pairs) - 1:03d}.png"))
        print(f"[{i + 1}/{len(picked)}] generated pair for caption {caption_index}")

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} pairs to {output_json}")


if __name__ == "__main__":
    main()
