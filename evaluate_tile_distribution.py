import argparse
import os
import torch
import random
import numpy as np
from collections import defaultdict, Counter
import json

from level_dataset import convert_to_level_format
import util.common_settings as common_settings
from captions.util import extract_tileset
from models.pipeline_loader import get_pipeline
from models.fdm_pipeline import FDMPipeline
from models.latent_diffusion_pipeline import UnconditionalDDPMPipeline
from tqdm.auto import tqdm

# TODO: Shouldn't this be moved to common_settings? Or is it basically already there?
# Per-game (tileset, tile_count, width, height) defaults. Mirrors the --game
# handling in train_diffusion.py so the two stay consistent.
_GAME_SETTINGS = {
    "Mario":     (common_settings.MARIO_TILESET, common_settings.MARIO_TILE_COUNT, common_settings.MARIO_WIDTH, common_settings.MARIO_HEIGHT),
    "MM2":       (common_settings.MM2_TILESET, common_settings.MM2_TILE_COUNT, common_settings.MM2_WIDTH, common_settings.MM2_HEIGHT),
    "LR":        (common_settings.LR_TILESET, common_settings.LR_TILE_COUNT, common_settings.LR_WIDTH, common_settings.LR_HEIGHT),
    "MM-Simple": (common_settings.MM_SIMPLE_TILESET, common_settings.MM_SIMPLE_TILE_COUNT, common_settings.MEGAMAN_WIDTH, common_settings.MEGAMAN_HEIGHT),
    "MM-Full":   (common_settings.MM_FULL_TILESET, common_settings.MM_FULL_TILE_COUNT, common_settings.MEGAMAN_WIDTH, common_settings.MEGAMAN_HEIGHT),
    "MMLV":      (common_settings.MMLV_TILESET, common_settings.MMLV_TILE_COUNT, common_settings.MEGAMAN_WIDTH, common_settings.MEGAMAN_HEIGHT),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate tile type distribution across a dataset and/or model checkpoints. "
                    "Tile names are derived entirely from the tileset's descriptors, so this "
                    "works for any game/tileset (original Mario, MM2, MM-Simple, MM-Full, Lode Runner)."
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model directory")
    parser.add_argument("--json_path", type=str, default=None, help="Path to a single dataset JSON file. Used alone to evaluate tile distribution for that dataset, or together with --epoch_dir as the training-set baseline for the comparison plot.")
    parser.add_argument("--epoch_dir", type=str, default=None, help="Path to a single trained checkpoint/epoch directory (e.g. a 'checkpoint-N' folder or final model dir). Generates samples from this checkpoint and compares their tile distribution and scene presence against --json_path, saving bar-chart visualizations of the difference.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of levels to generate per checkpoint")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--game", type=str, default="Mario", choices=["Mario", "MM2", "LR", "MM-Simple", "MM-Full", "MMLV"], help="Which game's tileset/level dimensions to use. Fills in --tileset, --num_tiles, --width and --height unless those are given explicitly.")
    parser.add_argument("--num_tiles", type=int, default=None, help="Number of tile types (defaults to the --game value)")
    parser.add_argument("--tileset", type=str, default=None, help="Path to tileset JSON (defaults to the --game value)")
    parser.add_argument("--width", type=int, default=None, help="Width of generated levels (defaults to the --game value)")
    parser.add_argument("--height", type=int, default=None, help="Height of generated levels (defaults to the --game value)")
    parser.add_argument("--inference_steps", type=int, default=common_settings.NUM_INFERENCE_STEPS, help="Denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=common_settings.GUIDANCE_SCALE, help="Classifier-free guidance scale")
    parser.add_argument("--captions_json", type=str, default=None, help="Path to a dataset JSON file (list of {\"caption\": ...} entries). If given, a fixed set of captions sampled from it is used to condition generation for every checkpoint, instead of generating unconditionally.")
    args = parser.parse_args()

    # Fill in tileset / tile count / dimensions from the chosen game, mirroring the
    # --game handling in train_diffusion.py. Explicit CLI values take precedence.
    tileset, tile_count, width, height = _GAME_SETTINGS[args.game]
    if args.tileset is None:
        args.tileset = tileset
    if args.num_tiles is None:
        args.num_tiles = tile_count
    if args.width is None:
        args.width = width
    if args.height is None:
        args.height = height

    if not args.model_path and not args.json_path and not args.epoch_dir:
        parser.error("Provide --model_path, --json_path, or --epoch_dir (with --json_path).")
    if args.model_path and args.epoch_dir:
        parser.error("Provide only one of --model_path or --epoch_dir, not both.")
    if args.model_path and args.json_path:
        parser.error("Provide only one of --model_path or --json_path, not both.")
    if args.epoch_dir and not args.json_path:
        parser.error("--epoch_dir requires --json_path (the training dataset to compare against).")

    return args


# Generic attribute descriptors shared across many tilesets. These describe a
# tile's behavior rather than its identity, so they make poor display names and
# are skipped when a more specific descriptor is available.
_SKIP_DESCRIPTORS = {
    "solid", "passable", "penetrable", "breakable", "movable", "moving",
    "damaging", "hazard", "enemy", "collectable", "powerup", "climbable",
    "empty", "null", "ranged", "jumping", "flying", "spawner", "spawn",
    "diggable", "pickupable", "interactive", "decoration", "vehicle",
    "projectile", "boss", "slope", "falling", "warp", "platform",
}


# Authoritative MM2 char→name mapping sourced from OBJ_META in MarioMakerPCG
# MM2's tileset descriptors are too generic to name enemies/items on their own
# (e.g. every enemy is just "enemy"/"moving"), so this supplies real names as a
# fallback when descriptors yield nothing specific. Only used for --game MM2.
_MM2_OBJ_NAMES = {
    # terrain
    "#": "Ground",          "B": "Block",               "H": "Hard Block",
    "?": "? Block",         "h": "Hidden Block",         "N": "Note Block",
    "d": "Donut Block",     "I": "Ice Block",            "p": "P Block",
    "O": "ON/OFF Block",    ".": "Dotted-Line Block",    "*": "Blinking Block",
    "Ç": "Spike Block",     "C": "Crate",                "S": "Stone",
    "{": "Starting Brick",  "=": "Castle Bridge",        "T": "Tree",
    "/": "Slight Slope",    "\\": "Steep Slope",
    # doors / warps
    "|": "Pipe",            "D": "Door",                 "W": "Warp Box",
    "k": "Key",             "f": "Checkpoint Flag",      "G": "Goal",
    "c": "Clear Pipe",
    # enemies
    "g": "Goomba",          "K": "Koopa",                "P": "Piranha Plant",
    "m": "Hammer Bro",      "t": "Thwomp",               "o": "Bob-omb",
    "s": "Spiny",           "b": "Buzzy Beetle",         "L": "Lakitu",
    "l": "Lakitu's Cloud",  "Z": "Banzai Bill",          "V": "Bullet Bill Blaster",
    "y": "Magikoopa",       "<": "Spike Top",            "u": "Boo",
    "X": "Bowser",          "x": "Bowser Jr.",           "@": "Chain Chomp",
    "~": "Cheep Cheep",     "q": "Blooper",              "w": "Wiggler",
    "Y": "Pokey",           "e": "Piranha Creeper",      "F": "Porcupuffer",
    "%": "Fish Bone",       "&": "Lava Bubble",          "r": "Rocky Wrench",
    ",": "Muncher",         "a": "Ant Trooper",          "n": "Monty Mole",
    "R": "Mechakoopa",      "!": "Boom Boom",            "9": "Dry Bones",
    "j": "Skipsqueak",      ";": "Stingby",              "A": "Angry Sun",
    "v": "Charvaargh",      "[": "Bully",
    "1": "Lemmy",           "2": "Morton",               "3": "Larry",
    "4": "Wendy",           "5": "Iggy",                 "6": "Roy",
    "7": "Ludwig",
    # items
    "¢": "Coin",            "$": "Red Coin",             "£": "Large Coin",
    "U": "1-Up Mushroom",   "i": "Fire Flower",          "¤": "Super Star",
    "M": "Super Mushroom",  "¶": "Big Mushroom",
    "¬": "Super Hammer",    "¦": "P Switch",             "¯": "POW Block",
    "±": "Spring",          "µ": "Goomba's Shoe",        "]": "Cannon Box",
    "}": "Propeller Box",   ")": "Goomba Mask",          "°": "Bullet Bill Mask",
    "²": "Red POW Box",
    # platforms
    "-": "Lift",            "³": "Mushroom Platform",    "´": "Semisolid Platform",
    "·": "Bridge",          "¸": "Lava Lift",            "¹": "Snake Block",
    "º": "Track Block",     "»": "Conveyor Belt",        "¼": "Fast Conveyor Belt",
    "½": "Sprint Platform", "¾": "Seesaw",               "¿": "Swinging Claw",
    "À": "ON/OFF Trampoline", "Á": "Mushroom Trampoline", "J": "Jumping Machine",
    "Â": "Half-Collision Platform", "Ã": "Donut",
    # hazards
    "Ä": "Fire Bar",        "Å": "Saw",                  "Æ": "Burner",
    "^": "Spikes",          "È": "Spike Ball",           "É": "Skewer",
    "Ê": "Twister",         "Ë": "Icicle",
    # decoration / other
    "Ì": "Cloud",           "Í": "Vine",                 "Î": "Water Marker",
    "Ï": "Arrow",           "Ð": "One-Way Wall",         "Ñ": "Reel Camera",
    "Ò": "Sound Effect",    "Ó": "Player",               "Ô": "Clown Car",
    "Õ": "Koopa Clown Car", "Ö": "Track",                "×": "Starting Arrow",
    "Ø": "Cannon",          "Ù": "! Block",
    # pipe directions (in tileset but not in OBJ_META)
    "↑": "Pipe (Up)",       "↓": "Pipe (Down)",
    "←": "Pipe (Left)",     "→": "Pipe (Right)",
}


def build_char_to_name(tileset_path, obj_names=None):
    """
    Return a dict mapping each tile char to a unique human-readable name derived
    purely from the tileset's own descriptors. Prefers the most specific,
    descriptive attribute (multi-word > non-generic single word > last listed).
    Any remaining duplicates are disambiguated by appending the char, and a tile
    with no usable descriptor falls back to the char itself.

    If `obj_names` (a char→name dict) is given, it is used as a fallback for any
    char whose descriptors yield nothing but a generic term — this is how MM2,
    whose enemy/item descriptors are too generic to name on their own, gets real
    names like "Goomba" instead of "enemy (g)".
    """
    with open(tileset_path, 'r', encoding='utf-8') as f:
        tileset = json.load(f)

    def pick_name_from_descriptors(descriptors):
        filtered = [d for d in descriptors if d not in ("solid", "passable")]
        if not filtered:
            return None
        multi_word = [d for d in filtered if ' ' in d]
        if multi_word:
            return max(multi_word, key=len)
        specific = [d for d in filtered if d not in _SKIP_DESCRIPTORS]
        if specific:
            return specific[-1]
        return filtered[-1]

    raw_names = {}
    for char, descriptors in tileset['tiles'].items():
        name = pick_name_from_descriptors(descriptors)
        if name and name not in _SKIP_DESCRIPTORS:
            raw_names[char] = name
        elif obj_names and char in obj_names:
            raw_names[char] = obj_names[char]
        else:
            raw_names[char] = name if name else char

    # Disambiguate: when two chars share a raw name, append (char) to both
    name_count = Counter(raw_names.values())
    char_to_name = {}
    for char, name in raw_names.items():
        if name_count[name] > 1:
            char_to_name[char] = f"{name} ({char})"
        else:
            char_to_name[char] = name

    return char_to_name


def generate_samples(pipe, num_samples, batch_size, inference_steps, guidance_scale, seed, height, width, captions=None):
    """Generate num_samples levels, returning a list of 2D tile-index scenes.

    If `captions` is given (a list of length num_samples), sample i is generated
    conditioned on captions[i]. Otherwise generation is unconditional.
    """
    is_unconditional = isinstance(pipe, UnconditionalDDPMPipeline)
    is_fdm = isinstance(pipe, FDMPipeline)

    if hasattr(pipe, 'unet') and pipe.unet is not None:
        device = next(pipe.unet.parameters()).device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_scenes = []
    remaining = num_samples
    offset = 0

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        generator = torch.Generator(device).manual_seed(seed)
        batch_captions = captions[offset:offset + current_batch] if captions is not None else None

        with torch.no_grad():
            if is_fdm:
                param_values = {"caption": batch_captions if batch_captions is not None else [""] * current_batch, "batch_size": current_batch}
            elif is_unconditional:
                param_values = {
                    "num_inference_steps": inference_steps,
                    "height": height, "width": width,
                    "output_type": "tensor", "batch_size": current_batch,
                }
            else:
                param_values = {
                    "num_inference_steps": inference_steps,
                    "height": height, "width": width,
                    "guidance_scale": guidance_scale,
                    "output_type": "tensor", "batch_size": current_batch,
                }
                if batch_captions is not None:
                    param_values["caption"] = batch_captions

            samples = pipe(generator=generator, **param_values).images

        for i in range(len(samples)):
            sample = samples[i].unsqueeze(0)
            scene = convert_to_level_format(sample)[0].tolist()
            all_scenes.append(scene)

        remaining -= current_batch
        offset += current_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_scenes


def count_tiles(scenes, id_to_char, char_to_name):
    """Count tile occurrences across all scenes, keyed by human-readable name."""
    counts = defaultdict(int)
    for scene in scenes:
        for row in scene:
            for tile_id in row:
                char = id_to_char.get(tile_id, '_')
                name = char_to_name.get(char, char)
                counts[name] += 1
    return dict(counts)


def count_scene_presence(scenes, id_to_char, char_to_name):
    """Count, for each tile name, how many scenes contain at least one occurrence
    of that tile (regardless of how many times it appears in that scene)."""
    counts = defaultdict(int)
    for scene in scenes:
        present = set()
        for row in scene:
            for tile_id in row:
                char = id_to_char.get(tile_id, '_')
                name = char_to_name.get(char, char)
                present.add(name)
        for name in present:
            counts[name] += 1
    return dict(counts)


def compute_distribution_stats(scenes, id_to_char, char_to_name):
    """Compute tile-count and scene-presence stats (raw counts and percentages)
    for a list of scenes."""
    counts = count_tiles(scenes, id_to_char, char_to_name)
    total = sum(counts.values())

    presence_counts = count_scene_presence(scenes, id_to_char, char_to_name)
    num_scenes = len(scenes)

    return {
        "num_scenes": num_scenes,
        "total_tiles": total,
        "tile_counts": counts,
        "tile_percentages": {n: 100.0 * c / total for n, c in counts.items()} if total else {},
        "scene_presence_counts": presence_counts,
        "scene_presence_percentages": {
            n: 100.0 * c / num_scenes for n, c in presence_counts.items()
        } if num_scenes else {},
    }


def load_captions_from_json(json_path):
    """Load caption strings from a dataset JSON file (list of dicts with a 'caption' key)."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [entry["caption"] for entry in data if isinstance(entry, dict) and "caption" in entry]


def load_scenes_from_json(json_path):
    """Load scenes from a dataset JSON file: a list of {"scene": [[...]], ...}
    entries, or a list of raw 2D scenes."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scenes = []
    for item in data:
        scenes.append(item["scene"] if isinstance(item, dict) else item)
    return scenes


def evaluate_json_dataset(json_path, id_to_char, char_to_name, tile_names):
    """Evaluate tile distribution and scene presence for a single dataset JSON
    file, writing the results to a single summary JSON file."""
    if not os.path.exists(json_path):
        print(f"Error: JSON file '{json_path}' does not exist.")
        return

    scenes = load_scenes_from_json(json_path)
    if not scenes:
        print(f"No scenes found in {json_path}")
        return

    stats = compute_distribution_stats(scenes, id_to_char, char_to_name)
    counts = stats["tile_counts"]
    total = stats["total_tiles"]
    presence_counts = stats["scene_presence_counts"]
    num_scenes = stats["num_scenes"]

    all_tiles = sorted(tile_names, key=lambda n: -counts.get(n, 0))

    result = {
        "source": json_path,
        "num_scenes": num_scenes,
        "total_tiles": total,
        "tile_counts": {n: counts.get(n, 0) for n in all_tiles},
        "tile_percentages": {n: round(stats["tile_percentages"].get(n, 0.0), 2) for n in all_tiles} if total else {n: 0.0 for n in all_tiles},
        "scene_presence_counts": {n: presence_counts.get(n, 0) for n in all_tiles},
        "scene_presence_percentages": {
            n: round(stats["scene_presence_percentages"].get(n, 0.0), 2) for n in all_tiles
        } if num_scenes else {n: 0.0 for n in all_tiles},
    }

    base, _ = os.path.splitext(json_path)
    out_path = f"{base}_tile_distribution.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    summary = "  ".join(
        f"{n}={counts.get(n, 0)} ({result['tile_percentages'][n]:.1f}%)"
        for n in all_tiles if counts.get(n, 0) > 0
    )
    print(f"\n{json_path} - Tile Distribution ({num_scenes} samples, {total} tiles)")
    print(f"  {summary}")
    print(f"\nDone. Results saved to: {out_path}")


def write_distribution(path, header, counts, total, tile_names):
    """Write a readable distribution file sorted by frequency."""
    present = [(n, counts.get(n, 0)) for n in tile_names]
    present.sort(key=lambda x: -x[1])
    col_width = max((len(n) for n, _ in present), default=10) + 2
    with open(path, 'w') as f:
        f.write(header + "\n")
        f.write("=" * (col_width + 30) + "\n")
        for name, count in present:
            pct = 100.0 * count / total if total > 0 else 0.0
            f.write(f"{name:<{col_width}}: {count:>8}  ({pct:5.1f}%)\n")
        f.write("=" * (col_width + 30) + "\n")
        f.write(f"{'Total':<{col_width}}: {total:>8}\n")


def plot_scene_presence(history, tile_names, out_path):
    """Plot the number of scenes containing each tile type, one line per tile,
    with epoch (checkpoint index) on the x-axis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not history:
        return

    epochs = [entry["epoch"] for entry in history]

    # Only plot tiles that show up in at least one checkpoint
    active_names = [
        name for name in tile_names
        if any(entry["presence"].get(name, 0) > 0 for entry in history)
    ]
    if not active_names:
        return

    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = plt.get_cmap("tab20")
    for i, name in enumerate(active_names):
        values = [entry["presence"].get(name, 0) for entry in history]
        ax.plot(epochs, values, label=name, color=cmap(i % 20), marker='o', markersize=3, linewidth=1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of scenes containing tile")
    ax.set_title("Scene-Level Tile Presence Across Checkpoints")
    ncol = max(1, (len(active_names) + 29) // 30)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize='x-small', ncol=ncol)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_bars(tile_names, model_values, dataset_values, title, ylabel, out_path):
    """Grouped bar chart with two bars per tile: one for the model output and
    one for the training dataset, side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not tile_names:
        return

    model_bars = [model_values.get(n, 0.0) for n in tile_names]
    dataset_bars = [dataset_values.get(n, 0.0) for n in tile_names]

    x = np.arange(len(tile_names))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(12, len(tile_names) * 0.5), 7))
    ax.bar(x - width / 2, model_bars, width, label='Model output', color='tab:blue')
    ax.bar(x + width / 2, dataset_bars, width, label='Training dataset', color='tab:orange')
    ax.set_xticks(x)
    ax.set_xticklabels(tile_names, rotation=90, fontsize='small')
    ax.set_xlim(-0.5, len(tile_names) - 0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def evaluate_epoch_vs_dataset(epoch_dir, json_path, id_to_char, char_to_name, tile_names, args, device):
    """Compare a single trained checkpoint's generated-sample tile distribution
    and scene presence against the training dataset's distribution, saving
    grouped bar charts (model vs. dataset, side by side per tile) for both
    metrics."""
    if not os.path.isdir(epoch_dir):
        print(f"Error: epoch directory '{epoch_dir}' does not exist.")
        return
    if not os.path.exists(json_path):
        print(f"Error: JSON file '{json_path}' does not exist.")
        return

    dataset_scenes = load_scenes_from_json(json_path)
    if not dataset_scenes:
        print(f"No scenes found in {json_path}")
        return
    dataset_stats = compute_distribution_stats(dataset_scenes, id_to_char, char_to_name)

    fixed_captions = None
    if args.captions_json:
        captions_pool = load_captions_from_json(args.captions_json)
        if captions_pool:
            if len(captions_pool) >= args.num_samples:
                fixed_captions = random.sample(captions_pool, args.num_samples)
            else:
                fixed_captions = random.choices(captions_pool, k=args.num_samples)
            print(f"Using {args.num_samples} fixed captions sampled from {args.captions_json}")

    print(f"Loading model from {epoch_dir}")
    pipe = get_pipeline(epoch_dir).to(device)
    model_type = "unconditional" if isinstance(pipe, UnconditionalDDPMPipeline) else "conditional"
    print(f"  Model type: {model_type}")

    model_scenes = generate_samples(
        pipe,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        height=args.height,
        width=args.width,
        captions=fixed_captions,
    )
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_stats = compute_distribution_stats(model_scenes, id_to_char, char_to_name)

    # Order tiles left-to-right by descending training-set frequency, then any
    # tiles that only appear in the model output by descending model frequency.
    ordered = sorted(
        (n for n in tile_names if dataset_stats["tile_counts"].get(n, 0) > 0),
        key=lambda n: -dataset_stats["tile_counts"][n],
    )
    leftover = sorted(
        (n for n in tile_names if n not in ordered and model_stats["tile_counts"].get(n, 0) > 0),
        key=lambda n: -model_stats["tile_counts"][n],
    )
    ordered += leftover
    if not ordered:
        print("No tiles found in either the dataset or the generated samples.")
        return

    out_dir = os.path.join(epoch_dir, "dataset_comparison")
    os.makedirs(out_dir, exist_ok=True)

    subtitle = f"({model_stats['num_scenes']} generated samples vs {dataset_stats['num_scenes']} dataset scenes)"

    dist_plot_path = os.path.join(out_dir, "distribution_diff.png")
    plot_comparison_bars(
        ordered,
        model_stats["tile_percentages"], dataset_stats["tile_percentages"],
        title=f"Tile Distribution: Model Output vs Training Dataset\n{subtitle}",
        ylabel="Model % - Dataset % (tile occurrences)",
        out_path=dist_plot_path,
    )

    presence_plot_path = os.path.join(out_dir, "presence_diff.png")
    plot_comparison_bars(
        ordered,
        model_stats["scene_presence_percentages"], dataset_stats["scene_presence_percentages"],
        title=f"Scene Presence: Model Output vs Training Dataset\n{subtitle}",
        ylabel="Model % - Dataset % (scenes containing tile)",
        out_path=presence_plot_path,
    )

    summary = {
        "epoch_dir": epoch_dir,
        "dataset_source": json_path,
        "model": {
            "num_scenes": model_stats["num_scenes"],
            "total_tiles": model_stats["total_tiles"],
            "tile_counts": {n: model_stats["tile_counts"].get(n, 0) for n in tile_names},
            "tile_percentages": {n: round(model_stats["tile_percentages"].get(n, 0.0), 2) for n in tile_names},
            "scene_presence_counts": {n: model_stats["scene_presence_counts"].get(n, 0) for n in tile_names},
            "scene_presence_percentages": {n: round(model_stats["scene_presence_percentages"].get(n, 0.0), 2) for n in tile_names},
        },
        "dataset": {
            "num_scenes": dataset_stats["num_scenes"],
            "total_tiles": dataset_stats["total_tiles"],
            "tile_counts": {n: dataset_stats["tile_counts"].get(n, 0) for n in tile_names},
            "tile_percentages": {n: round(dataset_stats["tile_percentages"].get(n, 0.0), 2) for n in tile_names},
            "scene_presence_counts": {n: dataset_stats["scene_presence_counts"].get(n, 0) for n in tile_names},
            "scene_presence_percentages": {n: round(dataset_stats["scene_presence_percentages"].get(n, 0.0), 2) for n in tile_names},
        },
    }
    summary_path = os.path.join(out_dir, "comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Comparison saved to: {out_dir}")


def collect_checkpoint_dirs(model_path):
    checkpoint_dirs = [
        (int(d.split("-")[-1]), os.path.join(model_path, d))
        for d in os.listdir(model_path)
        if os.path.isdir(os.path.join(model_path, d)) and d.startswith("checkpoint-")
    ]
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: x[0])

    has_unet = os.path.isdir(os.path.join(model_path, "unet"))
    has_fdm = os.path.isdir(os.path.join(model_path, "final-model")) or (
        not has_unet and any(
            f in os.listdir(model_path) for f in ["config.json", "pytorch_model.bin", "model.safetensors"]
        )
    )
    if checkpoint_dirs and (has_unet or has_fdm):
        checkpoint_dirs.append((checkpoint_dirs[-1][0] + 1, model_path))

    return checkpoint_dirs


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tile_chars, id_to_char, _, _ = extract_tileset(args.tileset)
    obj_names = _MM2_OBJ_NAMES if args.game == "MM2" else None
    char_to_name = build_char_to_name(args.tileset, obj_names)

    # Determine whether '_' is a real tile in this tileset or just the padding
    # sentinel appended by extract_tileset for tilesets that lack it.
    with open(args.tileset, 'r', encoding='utf-8') as _f:
        _underscore_is_padding = '_' not in json.load(_f)['tiles']

    # Unique ordered tile names — no duplicates; skip '_' only when it is the
    # padding sentinel (not a real tile in the JSON).
    seen = set()
    tile_names = []
    for c in tile_chars:
        if c == '_' and _underscore_is_padding:
            continue
        name = char_to_name.get(c, c)
        if name not in seen:
            seen.add(name)
            tile_names.append(name)

    if args.epoch_dir:
        evaluate_epoch_vs_dataset(args.epoch_dir, args.json_path, id_to_char, char_to_name, tile_names, args, device)
        return

    if args.json_path:
        evaluate_json_dataset(args.json_path, id_to_char, char_to_name, tile_names)
        return

    fixed_captions = None
    if args.captions_json:
        captions_pool = load_captions_from_json(args.captions_json)
        if not captions_pool:
            print(f"No captions found in {args.captions_json}")
            return
        if len(captions_pool) >= args.num_samples:
            fixed_captions = random.sample(captions_pool, args.num_samples)
        else:
            fixed_captions = random.choices(captions_pool, k=args.num_samples)
        print(f"Using {args.num_samples} fixed captions sampled from {args.captions_json} "
              f"(pool of {len(captions_pool)}) for every checkpoint")

    checkpoint_dirs = collect_checkpoint_dirs(args.model_path)
    if not checkpoint_dirs:
        print(f"No checkpoints found in {args.model_path}")
        return

    out_root = os.path.join(args.model_path, "tile_distributions")
    total_path = os.path.join(out_root, "total_distribution.txt")

    if os.path.exists(out_root):
        print(f"Error: Output directory '{out_root}' already exists. Delete it to re-run.")
        return

    created_dirs = []
    created_files = []

    try:
        os.makedirs(out_root)
        created_dirs.append(out_root)

        total_counts = defaultdict(int)
        total_tiles = 0

        presence_history = []
        presence_history_path = os.path.join(out_root, "scene_presence_history.json")
        created_files.append(presence_history_path)

        for epoch, checkpoint_dir in tqdm(checkpoint_dirs, desc="Evaluating checkpoints"):
            print(f"\nCheckpoint {epoch}: {checkpoint_dir}")
            pipe = get_pipeline(checkpoint_dir).to(device)
            model_type = "unconditional" if isinstance(pipe, UnconditionalDDPMPipeline) else "conditional"
            print(f"  Model type: {model_type}")

            scenes = generate_samples(
                pipe,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                height=args.height,
                width=args.width,
                captions=fixed_captions,
            )

            counts = count_tiles(scenes, id_to_char, char_to_name)
            total = sum(counts.values())

            presence_counts = count_scene_presence(scenes, id_to_char, char_to_name)
            num_scenes = len(scenes)

            # Per-checkpoint folder and file
            ckpt_dir = os.path.join(out_root, f"checkpoint-{epoch}")
            os.makedirs(ckpt_dir)
            created_dirs.append(ckpt_dir)

            dist_path = os.path.join(ckpt_dir, "distribution.txt")
            header = f"Checkpoint {epoch} — Tile Distribution ({num_scenes} samples, {total} tiles)"
            write_distribution(dist_path, header, counts, total, tile_names)
            created_files.append(dist_path)

            presence_path = os.path.join(ckpt_dir, "scene_presence.txt")
            presence_header = f"Checkpoint {epoch} — Scene Presence ({num_scenes} samples)"
            write_distribution(presence_path, presence_header, presence_counts, num_scenes, tile_names)
            created_files.append(presence_path)

            # Print quick summary to console
            summary = "  ".join(
                f"{n}={counts.get(n,0)} ({100*counts.get(n,0)/total:.1f}%)"
                for n in tile_names if counts.get(n, 0) > 0
            )
            print(f"  {summary}")

            for name, count in counts.items():
                total_counts[name] += count
            total_tiles += total

            presence_history.append({
                "epoch": epoch,
                "num_scenes": num_scenes,
                "presence": {n: presence_counts.get(n, 0) for n in tile_names},
            })
            with open(presence_history_path, 'w') as f:
                json.dump(presence_history, f, indent=2)

            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        header = f"Total Distribution Across All Checkpoints ({total_tiles} tiles)"
        write_distribution(total_path, header, total_counts, total_tiles, tile_names)
        created_files.append(total_path)

        plot_path = os.path.join(out_root, "scene_presence_plot.png")
        plot_scene_presence(presence_history, tile_names, plot_path)
        created_files.append(plot_path)

        print(f"\nDone. Results saved to: {out_root}")

    except Exception as e:
        print(f"\nError: {e}")
        print("Cleaning up output files created this run...")
        for path in created_files:
            if os.path.exists(path):
                os.remove(path)
                print(f"  Deleted: {path}")
        for d in reversed(created_dirs):
            try:
                os.rmdir(d)
                print(f"  Removed dir: {d}")
            except OSError:
                pass
        raise


if __name__ == "__main__":
    main()
