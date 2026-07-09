#!/usr/bin/env python3
"""

NEEDS TO BE MERGED WITH llm_ascii_to_caption.py


MarioMaker_llm_captions.py
==========================
Uses a local Ollama LLM to generate rich captions for Mario Maker 2
ASCII level scenes, replacing the simple tile-presence captions.

Supports resume: if the output file already exists, previously captioned
items are skipped and the run continues from where it left off.
Progress is saved every 10 new captions so an interrupted run loses minimal work.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
import urllib.error

# common_settings lives at the repo root; this script is under MM2_Files/, so make sure
# the repo root is importable whether it's run as a module or directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import util.common_settings as common_settings

# Reprompt budget for wrong caption counts. Kept large and separate from --max-reprompts
# (empty/non-English responses) since a lazy model usually complies on a later try.
MAX_CAPTION_RETRIES = 100

# ── Tile character → human-readable name ─────────────────────────────────────

EXTENDED_CHAR_NAMES = {
    "#": "Ground",
    "B": "Brick",
    "?": "Question Block",
    "c": "Coin",
    "g": "Enemy",
    "K": "Koopa",
    "P": "Piranha Plant",
    "t": "Thwomp",
    "^": "Spike",
    "N": "Block",
    "T": "Mushroom Platform",
    "=": "Bridge",
    "k": "Semisolid Platform",
    "S": "Stone",
    "i": "Fire Flower",
    "V": "Cannon",
    "|": "Pipe",
    "↑": "Pipe",
    "↓": "Pipe",
    "←": "Pipe",
    "→": "Pipe",
}

MM2_CHAR_NAMES = {
    " ": "Air",
    "#": "Ground",
    "B": "Brick Block",
    "H": "Hard Block",
    "?": "Question Block",
    "h": "Hidden Block",
    "N": "Note Block",
    "d": "Donut Block",
    "I": "Ice Block",
    "p": "P Block",
    "O": "On/Off Block",
    ".": "Dotted-Line Block",
    "*": "Blinking Block",
    "^": "Spike Block",
    "C": "Crate",
    "S": "Stone Block",
    "{": "Starting Brick",
    "=": "Castle Bridge",
    "T": "Tree",
    "/": "Slight Slope",
    "\\": "Steep Slope",
    "|": "Pipe",
    "↑": "Pipe (Up)",
    "↓": "Pipe (Down)",
    "←": "Pipe (Left)",
    "→": "Pipe (Right)",
    "D": "Door",
    "W": "Warp Box",
    "k": "Key",
    "f": "Checkpoint Flag",
    "G": "Goal",
    "c": "Clear Pipe",
    "g": "Goomba",
    "K": "Koopa Troopa",
    "P": "Piranha Plant",
    "m": "Hammer Bro",
    "t": "Thwomp",
    "o": "Bob-omb",
    "s": "Spiny",
    "b": "Buzzy Beetle",
    "L": "Lakitu",
    "l": "Lakitu's Cloud",
    "Z": "Banzai Bill",
    "V": "Bullet Bill Blaster",
    "y": "Magikoopa",
    "<": "Spike Top",
    "u": "Boo",
    "X": "Bowser",
    "x": "Bowser Jr.",
    "@": "Chain Chomp",
    "~": "Cheep Cheep",
    "q": "Blooper",
    "w": "Wiggler",
    "Y": "Pokey",
    "e": "Piranha Creeper",
    "F": "Porcupuffer",
    "%": "Fish Bone",
    "&": "Lava Bubble",
    "r": "Rocky Wrench",
    ",": "Muncher",
    "a": "Ant Trooper",
    "n": "Monty Mole",
    "R": "Mechakoopa",
    "!": "Boom Boom",
    "9": "Dry Bones",
    "j": "Skipsqueak",
    "+": "Cinobio",
    "\xa1": "Cinobic",
    ";": "Stingby",
    "A": "Angry Sun",
    "v": "Charvaargh",
    "[": "Bully",
    "1": "Lemmy Koopa",
    "2": "Morton Koopa Jr.",
    "3": "Larry Koopa",
    "4": "Wendy O. Koopa",
    "5": "Iggy Koopa",
    "6": "Roy Koopa",
    "7": "Ludwig von Koopa",
    # Style Ride slot (id 45): gamestyle-dependent (Yoshi's Egg in
    # SMW/NSMBU); scenes carry no gamestyle, so the SMB1/SMB3 name is
    # used here as a baseline. See mm2_json_field_dictionary.txt §6.
    "\xb5": "Goomba's Shoe",
    "\xa2": "Coin",
    "$": "Red Coin",
    "\xa3": "Big Coin",
    "U": "1-Up Mushroom",
    "i": "Fire Flower",
    "\xa4": "Super Star",
    "M": "Super Mushroom",
    # Style Power-up slot A: gamestyle-dependent (Super Leaf/Cape Feather/
    # Propeller Mushroom); scenes carry no gamestyle, so the SMB1 name is used
    # here as a baseline. See mm2_json_field_dictionary.txt §5. (The id-81 SMB2
    # Mushroom slot is dropped at extraction, so it has no glyph here.)
    "\xb6": "Big Mushroom",
    "\xac": "Super Hammer",
    "\xa6": "P Switch",
    "\xaf": "POW Block",
    "\xb1": "Spring",
    "]": "Cannon Box",
    "}": "Propeller Box",
    ")": "Goomba Mask",
    "\xb0": "Bullet Bill Mask",
    "\xb2": "Red POW Box",
    "-": "Lift",
    "\xb3": "Mushroom Platform",
    "\xb4": "Semisolid Platform",
    "\xb7": "Bridge",
    "\xb8": "Lava Lift",
    "\xb9": "Snake Block",
    "\xba": "Track Block",
    "\xbb": "Conveyor Belt",
    "\xbc": "Fast Conveyor Belt",
    "\xbd": "Sprint Platform",
    "\xbe": "Seesaw",
    "\xbf": "Swinging Claw",
    "\xc0": "On/Off Trampoline",
    "\xc1": "Trampoline",
    "J": "Jumping Machine",
    "\xc2": "Half-Collision Platform",
    "\xc3": "Donut Block Platform",
    "\xc4": "Fire Bar",
    "\xc5": "Saw",
    "\xc6": "Burner",
    "\xc7": "Spike Trap",
    "\xc8": "Spike Ball",
    "\xc9": "Skewer",
    "\xca": "Twister",
    "\xcb": "Icicle",
    "\xd8": "Cannon",
    "\xcc": "Cloud",
    "\xcd": "Vine",
    "\xce": "Water",
    "\xcf": "Arrow",
    "\xd0": "One-Way Wall",
    "\xd1": "Reel Camera",
    "\xd2": "Sound Effect",
    "\xd3": "Player Spawn",
    "\xd4": "Clown Car",
    "\xd5": "Koopa Car",
    "\xd6": "Track",
    "\xd7": "Starting Arrow",
    "\xd9": "Exclamation Block",
}

# ── Terrain character sets (solid tiles used for height/gap analysis) ─────────

TERRAIN_CHARS_MM2 = frozenset({"#", "H", "B", "S", "I", "C", "/", "\\"})
TERRAIN_CHARS_EXT = frozenset({"#", "B", "N", "S"})
TERRAIN_CHARS_WE = frozenset({"#", "B", "N", "?", "H", "I", "O"})

# ── Different games for their tilesets, right now MM2 and the three Mega Man tilesets ─────────


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "datasets")

GAME_SETTINGS = {
    "MM2": {
        "tileset": os.path.join(_REPO_ROOT, *common_settings.MM2_TILESET.split("/")),
        "tileset_we": os.path.join(_REPO_ROOT, *common_settings.MM2_TILESET.split("/")),
    },
    "MM": {
        "tileset": os.path.join(_DATASETS_DIR, "MM.json"),
        "tileset_we": os.path.join(_DATASETS_DIR, "MM.json"),
    },
    "MM-simple": {
        "tileset": os.path.join(_DATASETS_DIR, "MM-simple-tileset.json"),
        "tileset_we": os.path.join(_DATASETS_DIR, "MM-simple-tileset.json"),
    },
    "MMLV": {
        "tileset": os.path.join(_DATASETS_DIR, "MMLV.json"),
        "tileset_we": os.path.join(_DATASETS_DIR, "MMLV.json"),
    },
}

# ── Prompt template ───────────────────────────────────────────────────────────

_PROMPT_INTRO = """\
You are an expert Mario Maker 2 level describer.

You will receive three inputs:
1. A symbol dictionary mapping level grid symbols to Mario Maker 2 objects.
2. Pre-computed level metadata (RAW object tile counts, terrain column heights, floor/ceiling analysis, region boundaries).
3. A level grid (read top-to-bottom, left-to-right).

The metadata's object tile counts are raw occupied-cell counts, NOT deduplicated object counts. Many objects are single placed instances that render as a block of several identical cells — counting those cells directly will overcount how many objects are actually there. See MULTI-TILE OBJECTS below for how to interpret these counts and the grid correctly. Trust the metadata for terrain heights. Do not re-count terrain tiles from the grid, and never mention a feature that is not backed by the metadata or grid.

Your output trains a text-to-level diffusion model. The model needs many different human-style ways of describing the same level, so it can recognize a level from a short tag list, a casual sentence, or a detailed paragraph alike.

MULTI-TILE OBJECTS

Many object types occupy more than one grid cell per placed instance, appearing as a contiguous block of identical cells. Treat one contiguous block of the same tile type as ONE object, not one object per cell — unless the block's shape clearly looks like several same-size chunks repeated side by side, in which case count each repeated chunk as its own object.

This applies to (at least): pipes, bullet bill blasters, bridges, lifts, mushroom platforms, semisolid platforms, snake blocks, track blocks, conveyor belts, sprint platforms, half-collision platforms, donut block platforms, clouds, lava lifts, thwomps, angry suns, skewers, goal poles/flagpoles, vines, and fire bars.

If two blocks of the same tile type are not touching, they are separate objects. If you are uncertain whether a block is one object or several placed side by side, use your own knowledge of what sizes/dimensions that object can actually take in the Mario Maker 2 editor to judge the most plausible grouping, rather than guessing blindly. When still in doubt, prefer the more conservative (smaller) count.

WHAT YOU MAY DESCRIBE

Only describe concepts that are actually present in the metadata's object tile counts or terrain analysis, or that you can directly see in the grid using the symbol dictionary. Do not invent features. Concepts you may draw on, when present:

- terrain shape (floor, ceiling, gaps, hills, staircases, slopes, platforms, bridges, walls, pillars, towers, chambers, enclosed rooms, etc.) and its direction (ascending/descending) when relevant
- region structure (separate sections, dividers, left/center/right layout)
- block materials (ground, hard block, brick, ice block, note block, stone, etc.) when a structure is clearly made of one recognizable material
- pipes, doors, warps, and other traversal elements
- enemies and hazards, ideally with where they are (on the ground, on a platform, on a pipe, in the air, or which part of the level)
- collectibles and power-ups, ideally with placement
- rough quantities (one, two, three, a few, several, many) rather than exact tile counts for anything above three

You have free rein over how to organize and phrase this — which features to lead with, how to group them, and how much structural vocabulary to use. You do not need to follow a fixed checklist or priority order; just describe what's actually there in a way a person might.

WHAT TO AVOID

- Do not discuss gameplay, difficulty, quality, fun, or designer intent.
- Do not address the player or use second person.
- Do not describe exact coordinates or tile-by-tile detail.
- Do not mention features that are absent.

"""

# The task and output sections of the prompt depend on how many captions we
# want, so they are assembled per-run by build_prompt_template(). Everything
# after the WHAT TO AVOID section is appended dynamically below.

# Sample captions used only to show the model the JSON array shape. They are
# sliced to the requested caption count when building the example.
_EXAMPLE_CAPTIONS = [
    "ascending staircase. two goombas. one pipe right.",
    "a short level with a rising staircase, a couple of goombas, and a pipe near the end.",
    "the level begins on flat ground before climbing a set of steps toward the right side. two goombas patrol the lower section, and a pipe sits near the far right edge of the level.",
    "flat ground, a pipe, and a couple of enemies.",
    "rising terrain on the right with goombas below and a pipe at the far end.",
]


def _build_task_section(num_captions):
    if num_captions == 1:
        return (
            "YOUR TASK\n\n"
            "Write 1 caption for this level. It must be accurate and read like a natural human "
            "description. Make it a detailed, descriptive paragraph that walks through the level's "
            "layout and notable features in order. Keep it lowercase except for proper nouns "
            "inherent to object names if any."
        )
    return (
        "YOUR TASK\n\n"
        f"Write {num_captions} different captions for this same level. All {num_captions} must be "
        "accurate, but they must vary WIDELY from each other in length, level of detail, and "
        "register, so that together they cover the range of ways a human might describe this level:\n\n"
        "- one or two should be terse, tag-like phrases separated by periods (similar to keyword "
        "lists), covering only the 2-4 most prominent features\n"
        "- one or two should be a plain casual sentence or two, in normal prose, that a person "
        "might type quickly\n"
        "- one or two should be a more detailed, descriptive paragraph that walks through the "
        "level's layout and notable features in order\n\n"
        f"Across the {num_captions} captions, vary which features get emphasized — they don't all "
        "need to mention everything, but none should contradict another or invent something not "
        f"present. Keep all {num_captions} lowercase except for proper nouns inherent to object "
        "names if any."
    )


def _build_output_section(num_captions):
    if num_captions == 1:
        # The single-caption task asks for the detailed paragraph style, so show
        # that example rather than the terse tag-like first entry.
        entries = [_EXAMPLE_CAPTIONS[2]]
    else:
        n_example = min(num_captions, len(_EXAMPLE_CAPTIONS))
        entries = list(_EXAMPLE_CAPTIONS[:n_example])
        if num_captions > len(_EXAMPLE_CAPTIONS):
            entries += ["..."] * (num_captions - len(_EXAMPLE_CAPTIONS))
    example = json.dumps(entries, ensure_ascii=False)
    plural = "string" if num_captions == 1 else "strings"
    return (
        "OUTPUT FORMAT\n\n"
        f"Output a single JSON array of exactly {num_captions} {plural}, and nothing else — no "
        "markdown fences, no commentary, no keys other than the array itself.\n\n"
        "Example shape (do not reuse this content, it is only to show the format):\n"
        f"{example}"
    )


def build_system_section(num_captions):
    """The fixed, scene-independent part of the prompt (instructions/task/output).

    This is everything in build_prompt_template() before the per-scene
    dictionary/metadata/grid placeholders are appended.
    """
    return (
        _PROMPT_INTRO
        + _build_task_section(num_captions)
        + "\n\n"
        + _build_output_section(num_captions)
    )


def build_image_clause():
    """An instruction block, used only with --images-dir, explaining the rendered
    screenshot that accompanies the text so the model knows to use it."""
    return (
        "\n\nYou will also receive a rendered image of this same level region — a "
        "screenshot of how it looks in-game, covering the exact same area as the grid "
        "below. Use it to supplement the grid and metadata: disambiguate objects, confirm "
        "block materials, and read layout the symbols make unclear. The grid and metadata "
        "remain authoritative for object identity and counts — do not describe anything "
        "visible only in the image but absent from the grid/metadata, and never mention the "
        "image itself or that you were shown a picture."
    )


def build_prompt_template(num_captions, image_clause=""):
    """Assemble the full prompt template for the requested number of captions.

    Returns a string with {dict_string}, {metadata}, {grid_label}, and
    {ascii_grid} placeholders left intact for the per-scene .format() call.
    When image_clause is non-empty (--images-dir), it is woven into the
    instructions so the model knows a screenshot accompanies the text.
    """
    plural = "caption" if num_captions == 1 else "captions"
    return (
        build_system_section(num_captions)
        + image_clause
        + "\n\n"
        "Symbol dictionary:\n{dict_string}\n\n\n"
        "Metadata:\n{metadata}\n\n\n"
        "{grid_label}:\n{ascii_grid}\n\n"
        f"Write the JSON array of {num_captions} {plural} now. DO NOT INCLUDE ANY NON-ENGLISH "
        "CHARACTERS, and do not include anything outside the JSON array."
    )


def build_prompt_log_text(num_captions, dict_string, metadata, grid_label, ascii_grid,
                          image_clause=""):
    """Render the prompt as labeled triple-quoted blocks for a human-readable log file."""
    system_section = build_system_section(num_captions) + image_clause
    return (
        f'SYSTEM_PROMPT = """\n{system_section}\n"""\n\n'
        f'"""\n'
        f'Symbol dictionary:\n{dict_string}\n'
        f'"""\n\n'
        f'"""\n'
        f'Metadata:\n{metadata}\n'
        f'"""\n\n'
        f'"""\n'
        f'{grid_label}:\n\n{ascii_grid}\n\n'
        f'"""\n'
    )


# ── Core helpers ──────────────────────────────────────────────────────────────

def build_id_to_char(tileset_path):
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tile_chars = sorted(data["tiles"].keys(), key=lambda c: ord(c[0]))
    if "_" not in tile_chars:
        tile_chars.append("_")
    return {idx: char for idx, char in enumerate(tile_chars)}


# Category tags that describe how a tile behaves, not what it is. When
# picking a display name from a tileset's tag list, the last tag that is
# NOT one of these is used, since the truly identifying tag (e.g. "pipe")
# is not always written last (e.g. '|' = ["solid", "pipe", "warp"] would
# otherwise resolve to "Warp" instead of "Pipe").
GENERIC_TILE_TAGS = frozenset({
    "passable", "solid", "empty", "enemy", "damaging", "hazard", "moving",
    "flying", "boss", "explosive", "projectile", "falling", "slippery",
    "togglable", "collectable", "power-up", "platform", "vehicle",
    "style ride", "interactive", "climbable", "shooter",
    "warp", "decoration", "spawn",
})


def derive_char_names(tileset_path):
    """Build a char->name dict straight from a tileset's own tags.

    Used for tilesets (like mm2_tileset_we.json) that don't have a curated
    *_CHAR_NAMES dict, since their characters don't line up with MM2_CHAR_NAMES
    (e.g. "c" = coin here vs. "Clear Pipe" in mm2_tileset_full.json's char set).
    """
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = {}
    for char, tags in data["tiles"].items():
        if not tags:
            continue
        specific = [t for t in tags if t not in GENERIC_TILE_TAGS]
        chosen = specific[-1] if specific else tags[-1]
        names[char] = chosen.replace("-", " ").title()
    return names


def get_char_names(tileset_path):
    basename = os.path.basename(tileset_path)
    if "extended_tiles" in basename:
        return EXTENDED_CHAR_NAMES
    if basename == "mm2_tileset_full.json":
        return MM2_CHAR_NAMES
    # Everything else (the MM2 'we' set and the Mega Man tilesets) has no curated
    # name dict, so build names straight from each tileset's own tags.
    return derive_char_names(tileset_path)


def build_dict_string(tileset_path, char_names):
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    for char, tags in data["tiles"].items():
        name = char_names.get(char)
        if not name:
            name = tags[-1].replace("-", " ").title() if tags else "Unknown"
        lines.append(f"'{char}' = {name}")
    return "\n".join(lines)


def compute_metadata(scene, id_to_char, char_names, tileset_path):
    """Pre-compute level metadata to anchor the LLM's terrain and object descriptions."""
    if not scene or not scene[0]:
        return "No metadata available."

    grid = [[id_to_char.get(tid, " ") for tid in row] for row in scene]
    nrows = len(grid)
    ncols = len(grid[0])

    basename = os.path.basename(tileset_path)
    if "extended_tiles" in basename:
        terrain = TERRAIN_CHARS_EXT
    elif "tileset_we" in basename:
        terrain = TERRAIN_CHARS_WE
    else:
        terrain = TERRAIN_CHARS_MM2

    parts = []

    # Raw occupied-cell counts per tile type (skip Air and Ground). Multi-tile
    # objects are deduplicated in the prompt, not here — see MULTI-TILE OBJECTS.
    counts = {}
    for row in grid:
        for ch in row:
            if ch == " ":
                continue
            name = char_names.get(ch)
            if not name or name in ("Air", "Ground"):
                continue
            counts[name] = counts.get(name, 0) + 1

    if counts:
        parts.append("Object tile counts (raw occupied cells per type, see prompt notes on multi-tile objects):")
        for name, cnt in sorted(counts.items(), key=lambda x: -x[1])[:20]:
            parts.append(f"  {name}: {cnt}")

    # Terrain column height profile: for each column, height of topmost solid tile from bottom
    col_heights = []
    for c in range(ncols):
        h = 0
        for r in range(nrows - 1, -1, -1):
            if grid[r][c] in terrain:
                h = nrows - r  # 1 = bottom row, nrows = top row
                break
        col_heights.append(h)

    parts.append("\nTerrain top-of-column heights (left to right, 0=no terrain in column):")
    for start in range(0, ncols, 10):
        chunk = col_heights[start: start + 10]
        end = start + len(chunk)
        parts.append(f"  cols {start + 1:02d}-{end:02d}: {' '.join(str(h) for h in chunk)}")

    # Floor analysis: does the level have a continuous floor and where are the gaps?
    def col_has_floor(c):
        return any(grid[r][c] in terrain for r in range(nrows - 3, nrows))

    floor_mask = [col_has_floor(c) for c in range(ncols)]
    has_floor = sum(floor_mask) > ncols * 0.35

    gaps = []
    in_gap = False
    gap_start = 0
    for c in range(ncols):
        if not floor_mask[c] and not in_gap:
            in_gap = True
            gap_start = c + 1  # 1-indexed
        elif floor_mask[c] and in_gap:
            in_gap = False
            gaps.append(f"cols {gap_start}-{c}")
    if in_gap:
        gaps.append(f"cols {gap_start}-{ncols}")

    floor_str = "present" if has_floor else "absent"
    if gaps:
        floor_str += f", gaps at: {', '.join(gaps)}"
    parts.append(f"\nFloor: {floor_str}")

    # Ceiling analysis
    ceiling_count = sum(1 for c in range(ncols) if grid[0][c] in terrain)
    parts.append(f"Ceiling: {'present' if ceiling_count > ncols * 0.2 else 'absent'}")

    # Explicit region boundaries for position labeling
    t = ncols // 3
    parts.append(
        f"\nRegion boundaries (use these when assigning left/center/right):"
        f" left=cols 1-{t}, center=cols {t+1}-{2*t}, right=cols {2*t+1}-{ncols}"
    )

    return "\n".join(parts)


def scene_to_ascii(scene, id_to_char):
    return "\n".join(
        "".join(id_to_char.get(tid, "?") for tid in row)
        for row in scene
    )


# ── T0x token format (based on --tileset-we) ──────────────────────────────────

def build_char_to_token(id_to_char):
    """Map each tile character to a 'T<NN>' token, NN = its numeric tile ID."""
    width = max(2, len(str(len(id_to_char) - 1)))
    return {char: f"T{idx:0{width}d}" for idx, char in id_to_char.items()}


def build_token_dict_string(tileset_path, char_to_token, char_names):
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    for char, tags in data["tiles"].items():
        token = char_to_token[char]
        name = char_names.get(char)
        if not name:
            name = tags[-1].replace("-", " ").title() if tags else "Unknown"
        lines.append(f"{token} = {name}")
    return "\n".join(lines)


def scene_to_tokens(scene, id_to_char, char_to_token):
    unknown = char_to_token.get("?", "T??")
    return "\n".join(
        " ".join(char_to_token.get(id_to_char.get(tid, "?"), unknown) for tid in row)
        for row in scene
    )


def load_api_key(api_key_path):
    with open(api_key_path, "r", encoding="utf-8") as f:
        return f.readline().strip()


# ── Image input (--images-dir) ────────────────────────────────────────────────

_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Substrings identifying models that accept image input, used to guard
# --images-dir so an image isn't silently dropped against a text-only model
# (every current Claude model is multimodal; the local Ollama default is not).
VISION_MODEL_HINTS = (
    "claude",                                   # Anthropic (all current models)
    "gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-5",  # OpenAI
    "o1", "o3", "o4", "chatgpt",
    "gemini",                                   # Google
    "vision", "llava", "bakllava", "moondream",   # Ollama / open multimodal models
    "minicpm-v", "-vl", "vl-", "qwen2-vl", "qwen2.5vl", "qwen2.5-vl",
    "gemma3", "pixtral", "llama4", "mistral-small3",
    "smolvlm",                                  # HuggingFace SmolVLM / SmolVLM2
)


def model_supports_vision(model):
    m = model.lower()
    return any(hint in m for hint in VISION_MODEL_HINTS)


def load_image_b64(image_path):
    """Read an image and return (base64_string, media_type) for the API payload."""
    ext = os.path.splitext(image_path)[1].lower()
    media_type = _IMAGE_MEDIA_TYPES.get(ext, "image/png")
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def call_claude(prompt, model, api_key, max_tokens, timeout, retries,
                image_b64=None, media_type=None):
    if image_b64:
        content = [
            {"type": "image", "source": {
                "type": "base64", "media_type": media_type, "data": image_b64}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                parts = result.get("content", [])
                return "".join(p.get("text", "") for p in parts).strip()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"  [RETRY {attempt + 1}/{retries - 1}] {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Claude request failed after {retries} attempts: {e}"
                ) from e


def call_openai(prompt, model, api_key, max_tokens, timeout, retries,
                image_b64=None, media_type=None):
    if image_b64:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:{media_type};base64,{image_b64}"}},
        ]
    else:
        content = prompt
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                choices = result.get("choices", [])
                if not choices:
                    return ""
                return (choices[0].get("message", {}).get("content") or "").strip()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"  [RETRY {attempt + 1}/{retries - 1}] {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"OpenAI request failed after {retries} attempts: {e}"
                ) from e


def call_gemini(prompt, model, api_key, max_tokens, timeout, retries,
                image_b64=None, media_type=None):
    parts = []
    if image_b64:
        parts.append({"inline_data": {"mime_type": media_type, "data": image_b64}})
    parts.append({"text": prompt})
    payload = json.dumps({
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
        },
    }).encode("utf-8")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if not candidates:
                    return ""
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts).strip()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"  [RETRY {attempt + 1}/{retries - 1}] {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Gemini request failed after {retries} attempts: {e}"
                ) from e


def call_ollama(prompt, model, url, timeout, retries, max_tokens=900,
                num_ctx=8192, temperature=0.4, image_b64=None, media_type=None):
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Thinking models (e.g. the gemma the reference captions with) otherwise
        # spend the whole generation budget on a reasoning trace and return an
        # empty "response" -- or, with num_predict set, never reach a final answer
        # at all. Disabling thinking sends every token to the caption itself, which
        # is exactly what the working MarioDiffusion code does (ollama.chat
        # think=False). Non-thinking models accept and ignore this.
        "think": False,
        "options": {
            # A small touch of temperature avoids the degenerate empty-output
            # collapse gemma-family models fall into under fully greedy decoding.
            # num_ctx must hold the whole prompt (dictionary + metadata +
            # space-separated T## grid) AND leave room to generate, but setting it
            # too high blows the KV cache past VRAM on a 12B+ model and Ollama then
            # returns an empty response -- so it is tunable via --num-ctx.
            # num_predict caps output the way --max-tokens does for the API backends.
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": max_tokens,
        },
    }
    if image_b64:
        # The /api/generate endpoint takes raw base64 strings (no data: prefix).
        body["images"] = [image_b64]
    payload = json.dumps(body).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result.get("response", "").strip()
                if not text:
                    # Surface Ollama's own accounting so an empty generation is
                    # diagnosable rather than silent. prompt_tokens approaching
                    # num_ctx means the prompt overflowed the window; output_tokens
                    # of 0 with done_reason 'load' points at a failed model/KV-cache
                    # load (often num_ctx too large for VRAM), while done_reason
                    # 'stop' with 0 output is a greedy/template collapse.
                    print(
                        f"\n    [ollama empty] done_reason={result.get('done_reason')!r} "
                        f"prompt_tokens={result.get('prompt_eval_count')} "
                        f"output_tokens={result.get('eval_count')} num_ctx={num_ctx}",
                        end="", flush=True,
                    )
                return text
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"  [RETRY {attempt + 1}/{retries - 1}] {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Ollama request failed after {retries} attempts: {e}"
                ) from e


def _ollama_base_url(generate_url):
    """Strip the '/api/...' path off the Ollama endpoint to get its server root.

    The user passes the /api/generate URL; /api/tags and /api/pull live on the
    same server, so derive them from that instead of adding more flags.
    """
    marker = "/api/"
    idx = generate_url.find(marker)
    return generate_url[:idx] if idx != -1 else generate_url.rstrip("/")


def ollama_model_available(model, generate_url, timeout=10):
    """Return True if `model` is already pulled on the Ollama server.

    Ollama tags carry an implicit ':latest' when no tag is given, so a bare
    'llava' should match the listed 'llava:latest'. Any error talking to the
    server returns False so the caller falls through to a pull attempt (which
    surfaces a clearer error if the server is truly down).
    """
    url = _ollama_base_url(generate_url) + "/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return False
    names = {m.get("name", "") for m in data.get("models", [])}
    if model in names:
        return True
    # Match a bare name against its ':latest' tag and vice versa.
    wanted = model if ":" in model else f"{model}:latest"
    return wanted in names


def ollama_pull_model(model, generate_url, timeout=1800):
    """Pull `model` onto the Ollama server, streaming progress to the console.

    Mirrors `ollama pull <model>`: the /api/pull endpoint streams newline-
    delimited JSON status objects, some with completed/total byte counts. We
    render a single updating percentage line so a multi-GB download shows life.
    Raises RuntimeError on failure (unknown model, network drop, disk full).
    """
    url = _ollama_base_url(generate_url) + "/api/pull"
    payload = json.dumps({"model": model, "stream": True}).encode("utf-8")
    print(f"Model '{model}' not found on the Ollama server; pulling it now "
          f"(this can take a while the first time)...", flush=True)
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    last_status = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if msg.get("error"):
                    raise RuntimeError(f"Ollama pull failed: {msg['error']}")
                status = msg.get("status", "")
                total = msg.get("total")
                completed = msg.get("completed")
                if total and completed is not None:
                    pct = completed / total * 100
                    print(f"\r  {status}: {pct:5.1f}% "
                          f"({completed / 1e9:.2f}/{total / 1e9:.2f} GB)   ",
                          end="", flush=True)
                elif status and status != last_status:
                    # A phase change with no byte counts (manifest, verify, etc.).
                    print(f"\r  {status}" + " " * 40, end="", flush=True)
                last_status = status
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"Ollama pull request failed: {e}") from e
    print(f"\r  pull complete: {model}" + " " * 40, flush=True)


def ensure_ollama_model(model, generate_url):
    """Make sure `model` is present on the Ollama server, pulling it if not.

    Called once before captioning starts for the ollama/moondream/llava
    backends, so a missing model is downloaded up front instead of every
    /api/generate call 404ing into the retry/backoff loop.
    """
    if ollama_model_available(model, generate_url):
        return
    ollama_pull_model(model, generate_url)


# SmolVLM is a local HuggingFace model loaded in-process (not an HTTP API like
# the other backends). Loading it is expensive, so the processor/model are cached
# here and reused across every scene in a run; only the first call pays the cost.
_SMOLVLM_STATE = {}


def call_smolvlm(prompt, model, max_tokens, image_b64=None, media_type=None):
    """Run a local HuggingFace SmolVLM / SmolVLM2 model on the prompt (+ image).

    Unlike the API backends, this loads weights into local memory via transformers
    and runs generation in-process. The model is loaded once and cached in
    _SMOLVLM_STATE, keyed by model name, so repeated calls reuse it.
    """
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError as e:
        raise RuntimeError(
            "The smolvlm backend needs torch, transformers, and Pillow:\n"
            "    pip install torch transformers pillow\n"
            f"(import failed: {e})"
        ) from e

    if _SMOLVLM_STATE.get("model_name") != model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"  [smolvlm] loading {model} on {device} ...", flush=True)
        processor = AutoProcessor.from_pretrained(model)
        vlm = AutoModelForImageTextToText.from_pretrained(
            model, torch_dtype=dtype
        ).to(device)
        _SMOLVLM_STATE.update(
            model_name=model, processor=processor, model=vlm, device=device
        )

    processor = _SMOLVLM_STATE["processor"]
    vlm = _SMOLVLM_STATE["model"]
    device = _SMOLVLM_STATE["device"]

    images = []
    content = []
    if image_b64:
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        images.append(img)
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=chat, images=images or None, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        # do_sample=False -> greedy/deterministic, matching the temperature=0 the
        # API backends use.
        gen = vlm.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    decoded = processor.batch_decode(
        gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return decoded[0].strip() if decoded else ""


def find_non_ascii_chars(captions):
    """Return a sorted list of all distinct non-ASCII characters across captions.

    Returns an empty list if every character is plain ASCII (code points 0-127).
    """
    bad = {ch for caption in captions for ch in caption if ord(ch) > 127}
    return sorted(bad)


def build_avoidance_clause(bad_chars):
    """Build a CRITICAL clause banning every character seen so far this run.

    Returns "" when no bad characters have been collected yet, so a clean run
    leaves the base prompt untouched.
    """
    if not bad_chars:
        return ""
    char_list = ", ".join(f"{ch!r} (U+{ord(ch):04X})" for ch in sorted(bad_chars))
    return (
        f"\n\nCRITICAL: The following non-ASCII characters have appeared in earlier outputs during "
        f"this run and are strictly forbidden: {char_list}. NEVER use any of them. Use only plain "
        f"ASCII characters (code points 0-127) — for example, a hyphen-minus '-' instead of an em "
        f"dash '—', and straight quotes instead of curly quotes. Output the JSON array with only "
        f"ASCII characters."
    )


def build_count_clause(num_captions, retries):
    """Clause re-stressing the exact caption count, added once a scene has miscounted.

    Escalates with the retry number: a firm reminder for the first couple of misses,
    then a blunter demand plus an explicit N-slot skeleton once the model keeps
    miscounting, since a lazy model usually needs a harder shove than a polite
    restatement. Returns "" until the scene has miscounted at least once, like
    build_avoidance_clause for non-ASCII chars.
    """
    if not retries:
        return ""
    plural = "caption" if num_captions == 1 else "captions"
    string_plural = "string" if num_captions == 1 else "strings"
    base = (
        f"\n\nCRITICAL: An earlier attempt for this level returned the WRONG number of captions. You "
        f"MUST output EXACTLY {num_captions} {plural} -- no more and no fewer -- as a single JSON array "
        f"of {num_captions} {string_plural}. Returning a different number (for example only one when "
        f"more are asked for) is a failure: count your captions before answering and make sure there "
        f"are exactly {num_captions}."
    )
    if retries < 3:
        return base
    # Harder shove after repeated miscounts: spell out the exact array shape with one
    # slot per caption so the model only has to fill blanks, and forbid any stray text.
    slots = ", ".join(f'"<caption {i + 1}>"' for i in range(num_captions))
    return base + (
        f"\n\nThis has now failed {retries} times. STOP and follow this exactly. Output ONLY a JSON "
        f"array with EXACTLY {num_captions} elements, in this shape, replacing each placeholder with a "
        f"real caption while keeping the brackets and commas:\n[{slots}]\n"
        f"There must be exactly {num_captions} comma-separated {string_plural} between the brackets and "
        f"absolutely nothing before '[' or after ']'."
    )


# Conservative chars-per-token ratios, measured against Ollama's prompt_eval_count
# on this machine (gemma3). Each is the LOW end of the observed range for that prompt
# shape, so dividing by it overestimates the token count and the fitted context
# window errs toward leaving enough room. The space-separated 'T<NN>' token grid is
# far denser (~1.5-2.0 chars/token) than the character ASCII grid + prose (~4-5).
_CHARS_PER_TOKEN = {"tokens": 1.5, "ascii": 3.5}


def estimate_prompt_tokens(text, grid_format="ascii"):
    """Conservative (deliberately high) token estimate for an Ollama prompt."""
    return int(len(text) / _CHARS_PER_TOKEN.get(grid_format, 3.5)) + 1


def fit_num_ctx(prompt, floor_ctx, max_ctx, gen_tokens, grid_format="ascii"):
    """Pick an Ollama num_ctx that holds the prompt AND leaves room to generate.

    Grows floor_ctx (the user's --num-ctx) up to max_ctx when a big prompt -- e.g.
    the space-separated token grid on a wide level -- would otherwise fill the whole
    window and starve generation. That starvation is what makes Ollama return an
    empty or truncated response (done_reason 'length', ~0 output tokens), which the
    caller then reprompts against forever since every retry sends the same oversized
    prompt. Returns (effective_ctx, fits); fits is False when even max_ctx cannot
    seat the prompt with meaningful room to answer.
    """
    prompt_tokens = estimate_prompt_tokens(prompt, grid_format)
    needed = prompt_tokens + gen_tokens + 512  # 512 ~ headroom for reprompt clauses
    effective = min(max(floor_ctx, needed), max_ctx)
    # Require at least a little generation room beyond the prompt itself.
    fits = prompt_tokens + 256 <= max_ctx
    return effective, fits

_UNICODE_NORMALIZE = {
    '\u2018': "'", '\u2019': "'",   # left/right single quotes
    '\u201c': '"', '\u201d': '"',   # left/right double quotes
    '\u2014': '-', '\u2013': '-',   # em dash, en dash
    '\u2026': '...',                # ellipsis
    '\u00e9': 'e', '\u00e8': 'e',  # accented e (common in French loanwords)
    '\u00e0': 'a', '\u00e2': 'a',  # accented a
    '\u00f4': 'o',                  # accented o
    '\u00fc': 'u', '\u00fb': 'u',  # accented u
    '\u00b7': '.', '\u2022': '.',   # middle dot, bullet
    '\u2019': "'",                  # right single quote (duplicate for safety)
}

def normalize_to_ascii(text):
    """Replace common non-ASCII characters with ASCII equivalents."""
    return ''.join(_UNICODE_NORMALIZE.get(ch, ch) for ch in text)


def parse_captions(raw_response):
    """Parse the LLM's JSON array of captions, with a line-based fallback.

    Returns a list of caption strings (possibly empty if parsing fails entirely).
    """
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return [normalize_to_ascii(c).strip() for c in parsed if c.strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: try to find the first [...] block in the response.
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return [c.strip() for c in parsed if c.strip()]
        except json.JSONDecodeError:
            pass

    # Last resort: treat each non-empty line as its own caption.
    lines = [ln.strip().lstrip("0123456789.-) ").strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def captions_from_entry(entry):
    """Collect an entry's captions from 'caption', 'caption1', 'caption2', ...

    Falls back to the older 'captions' list field for datasets generated
    before this naming switch, so resume still works on older output files.
    """
    captions = [entry["caption"]] if entry.get("caption") else []
    idx = 1
    while f"caption{idx}" in entry:
        captions.append(entry[f"caption{idx}"])
        idx += 1
    if not captions and entry.get("captions"):
        captions = list(entry["captions"])
    return captions


def load_existing(output_path, caption_mode="legacy", caption_key=None):
    if not os.path.isfile(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if caption_mode == "keyed":
        return {item["name"]: item for item in data if caption_key in item}
    return {
        item["name"]: item for item in data
        if "captions" in item or "caption" in item or "caption1" in item
    }


def _write(output_path, data):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _validate_tileset_match(dataset, id_to_char, tileset_path):
    """Abort early if the dataset's tile IDs don't fit the loaded tileset.

    A common mistake is pairing a dataset built with extended_tiles.json (max ID ~22)
    against mm2_tileset_full.json (138 tiles, different sort order), or vice versa.
    When the IDs don't match, the ASCII fed to the LLM is completely wrong.
    """
    tileset_size = len(id_to_char)
    sample_count = min(50, len(dataset))
    max_seen = 0
    unknown_count = 0

    for item in dataset[:sample_count]:
        scene = item["scene"] if isinstance(item, dict) else item
        for row in scene:
            for tid in row:
                if tid > max_seen:
                    max_seen = tid
                if tid not in id_to_char:
                    unknown_count += 1

    if max_seen >= tileset_size:
        print(
            f"\nERROR: Tileset mismatch detected!\n"
            f"  Tileset '{os.path.basename(tileset_path)}' has {tileset_size} tiles (IDs 0-{tileset_size-1}).\n"
            f"  Dataset contains tile ID {max_seen}, which is out of range.\n"
            f"  You are probably using the wrong --tileset for this dataset.\n"
            f"  If the dataset was built with extended_tiles.json, pass --tileset extended_tiles.json.\n"
            f"  If the dataset was built with mm2_tileset_full.json, pass --tileset mm2_tileset_full.json.\n"
        )
        sys.exit(1)

    if unknown_count > 0:
        print(
            f"WARNING: {unknown_count} tile IDs in the first {sample_count} scenes "
            f"have no mapping in the tileset. The ASCII grid may contain '?' characters."
        )


def generate_captions(dataset_path, tileset_path, output_path, model, url, timeout, retries,
                       grid_format="ascii", tileset_we_path=None, ascii_output_dir=None,
                       backend="ollama", api_key=None, max_tokens=900, max_reprompts=3,
                       num_captions=5, prompt_log_file="MM2_Prompt.txt",
                       num_ctx=8192, max_num_ctx=16384, temperature=0.4, images_dir=None,
                       caption_mode="legacy", caption_key=None):
    if caption_mode == "keyed" and not caption_key:
        caption_key = f"{model}_captions"  # default the source key to the model name

    # A vision-capable model always gets images alongside the grid; a
    # text-only model never does. There's no separate flag for this.
    with_images = model_supports_vision(model)
    image_clause = build_image_clause() if with_images else ""
    prompt_template = build_prompt_template(num_captions, image_clause)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Each dataset item's 'image' path (written by mm2pipeline_data.dataset build
    # --with_images) is resolved against --images-dir, defaulting to the
    # dataset JSON's own folder.
    dataset_dir = os.path.abspath(images_dir) if images_dir else os.path.dirname(os.path.abspath(dataset_path))
    if with_images:
        has_any_image = any(
            isinstance(it, dict) and it.get("image") for it in dataset
        )
        if not has_any_image:
            print(
                f"ERROR: model '{model}' is vision-capable, but no items in the dataset carry "
                "an 'image' path.\n  Rebuild the dataset with 'python -m mm2pipeline_data dataset "
                "build --with_images' so each sample gets a cropped PNG, then retry."
            )
            sys.exit(1)

    # For the local Ollama-served backends, make sure the model is pulled before
    # the run starts. Otherwise every /api/generate call 404s and disappears into
    # the retry/backoff loop, which never recovers (a missing model isn't
    # transient). This mirrors doing `ollama pull <model>` by hand first.
    if backend in ("ollama", "moondream", "llava"):
        ensure_ollama_model(model, url)

    id_to_char = build_id_to_char(tileset_path)
    char_names = get_char_names(tileset_path)

    _validate_tileset_match(dataset, id_to_char, tileset_path)

    if grid_format == "tokens":
        we_id_to_char = build_id_to_char(tileset_we_path)
        char_to_token = build_char_to_token(we_id_to_char)
        we_char_names = get_char_names(tileset_we_path)
        dict_string = build_token_dict_string(tileset_we_path, char_to_token, we_char_names)
        grid_label = "Token Grid (each cell is a tile-ID token, space-separated)"
    else:
        dict_string = build_dict_string(tileset_path, char_names)
        grid_label = "ASCII Level"

    if ascii_output_dir:
        os.makedirs(ascii_output_dir, exist_ok=True)

    existing = load_existing(output_path, caption_mode, caption_key)
    if existing:
        print(f"Resuming: {len(existing)} captions already present in {output_path}")
    else:
        print("Starting fresh.")

    if caption_mode == "keyed":
        print(f"Keyed mode: writing captions under '{caption_key}'")

    results = []
    total = len(dataset)
    generated = 0
    skipped = 0
    errors = 0
    images_used = 0
    images_missing = 0
    all_bad_chars = set()
    total_reprompts = 0
    reprompted_scenes = {}  # scene name -> number of reprompts it needed
    last_full_prompt = None  # most recent fully-rendered prompt (dict + metadata + grid substituted in)
    last_metadata = None
    last_ascii_grid = None

    for i, item in enumerate(dataset):
        scene = item["scene"] if isinstance(item, dict) else item
        name = item.get("name", str(i)) if isinstance(item, dict) else str(i)

        if grid_format == "tokens":
            ascii_grid = scene_to_tokens(scene, id_to_char, char_to_token)
        else:
            ascii_grid = scene_to_ascii(scene, id_to_char)

        if ascii_output_dir:
            safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
            with open(os.path.join(ascii_output_dir, f"{safe_name}.txt"), "w", encoding="utf-8") as f:
                f.write(ascii_grid)

        if name in existing:
            cached = existing[name]
            # Older outputs store captions under "captions" (a list); migrate them
            # to "caption"/"caption1"/"caption2"/... so resumed runs use the
            # current on-disk shape.
            if "caption" not in cached and cached.get("captions"):
                migrated = captions_from_entry(cached)
                cached.pop("captions", None)
                for idx, cap in enumerate(migrated):
                    cached["caption" if idx == 0 else f"caption{idx}"] = cap
            results.append(cached)
            skipped += 1
            continue

        metadata = compute_metadata(scene, id_to_char, char_names, tileset_path)
        prompt = prompt_template.format(
            dict_string=dict_string,
            grid_label=grid_label,
            ascii_grid=ascii_grid,
            metadata=metadata,
        )
        last_metadata = metadata
        last_ascii_grid = ascii_grid

        # Resolve this scene's rendered image once (reused across reprompts).
        image_b64 = None
        media_type = None
        image_tag = ""
        if with_images:
            image_rel = item.get("image") if isinstance(item, dict) else None
            if image_rel:
                image_path = os.path.join(dataset_dir, image_rel)
                if os.path.isfile(image_path):
                    image_b64, media_type = load_image_b64(image_path)
                    image_tag = " [img]"
                    images_used += 1
                else:
                    image_tag = " [img missing]"
                    images_missing += 1
            else:
                image_tag = " [no img]"
                images_missing += 1

        # Size the Ollama context window to this scene so a big prompt (notably the
        # space-separated token grid on a wide level) can't fill the whole window and
        # leave nothing to generate -- the overflow that otherwise produces empty/
        # truncated output and an endless reprompt loop. Other backends manage their
        # own context, so this only applies to the local Ollama path.
        effective_num_ctx = num_ctx
        if backend in ("ollama", "moondream", "llava"):
            effective_num_ctx, fits = fit_num_ctx(
                prompt, num_ctx, max_num_ctx, max_tokens, grid_format)
            if not fits:
                print(
                    f"[{i + 1}/{total}] {name}{image_tag} ... SKIP: prompt needs "
                    f"~{estimate_prompt_tokens(prompt, grid_format)} tokens, more than "
                    f"--max-num-ctx ({max_num_ctx}) can hold with room to generate. Use "
                    f"--grid-format ascii (far fewer tokens than the token grid) or raise "
                    f"--max-num-ctx."
                )
                errors += 1
                continue

        ctx_tag = f" [ctx {effective_num_ctx}]" if effective_num_ctx != num_ctx else ""
        print(f"[{i + 1}/{total}] {name}{image_tag}{ctx_tag} ...", end=" ", flush=True)
        captions = []
        last_raw = ""
        start_time = time.time()
        # attempt = empty/non-English reprompts (max_reprompts);
        # caption_retries = wrong-count reprompts (MAX_CAPTION_RETRIES).
        attempt = 0
        caption_retries = 0
        try:
            while attempt < max_reprompts and caption_retries < MAX_CAPTION_RETRIES:
                # Carry the running bad-char ban list, plus a count reminder once this
                # scene has miscounted.
                active_prompt = (
                    prompt
                    + build_avoidance_clause(all_bad_chars)
                    + build_count_clause(num_captions, caption_retries)
                )
                last_full_prompt = active_prompt

                if backend == "claude":
                    raw = call_claude(active_prompt, model, api_key, max_tokens, timeout,
                                      retries, image_b64=image_b64, media_type=media_type)
                elif backend == "openai":
                    raw = call_openai(active_prompt, model, api_key, max_tokens, timeout,
                                      retries, image_b64=image_b64, media_type=media_type)
                elif backend == "gemini":
                    raw = call_gemini(active_prompt, model, api_key, max_tokens, timeout,
                                      retries, image_b64=image_b64, media_type=media_type)
                elif backend == "smolvlm":
                    raw = call_smolvlm(active_prompt, model, max_tokens,
                                       image_b64=image_b64, media_type=media_type)
                else:
                    # ollama, moondream and llava all go through the local Ollama
                    # server; moondream/llava just pin a particular vision model.
                    raw = call_ollama(active_prompt, model, url, timeout, retries,
                                      max_tokens=max_tokens, num_ctx=effective_num_ctx,
                                      temperature=temperature,
                                      image_b64=image_b64, media_type=media_type)
                captions = parse_captions(raw)
                last_raw = raw

                # An empty/unparseable response yields no captions. Reprompt
                # instead of breaking out and dropping the level -- gemma
                # occasionally returns "" for a scene that succeeds on a retry.
                if not captions:
                    attempt += 1
                    reprompted_scenes[name] = reprompted_scenes.get(name, 0) + 1
                    total_reprompts += 1
                    print(
                        f"\n  [REPROMPT {attempt}/{max_reprompts - 1}] "
                        f"empty response, retrying...",
                        end=" ", flush=True,
                    )
                    continue

                bad_chars = find_non_ascii_chars(captions)
                if bad_chars:
                    all_bad_chars.update(bad_chars)
                    attempt += 1
                    reprompted_scenes[name] = reprompted_scenes.get(name, 0) + 1
                    total_reprompts += 1
                    current_list = ", ".join(f"{ch!r} (U+{ord(ch):04X})" for ch in bad_chars)
                    print(
                        f"\n  [REPROMPT {attempt}/{max_reprompts - 1}] "
                        f"non-English character(s) {current_list} in caption, retrying...",
                        end=" ", flush=True,
                    )
                    captions = []
                    continue

                # Gemma especially under-produces here; reprompt until the count matches.
                if len(captions) != num_captions:
                    caption_retries += 1
                    reprompted_scenes[name] = reprompted_scenes.get(name, 0) + 1
                    total_reprompts += 1
                    print(
                        f"\n  [CAPTION RETRY {caption_retries}/{MAX_CAPTION_RETRIES}] "
                        f"returned {len(captions)} caption(s), expected {num_captions}; "
                        f"retrying...",
                        end=" ", flush=True,
                    )
                    captions = []
                    continue

                break
        except RuntimeError as e:
            print(f"ERROR: {e}")
            captions = []

        if not captions:
            print("ERROR: no usable captions after reprompts; dropping level from output")
            print(f"  Last raw response: {last_raw!r}")  # ADD THIS
            errors += 1
            continue

        elapsed = time.time() - start_time
        print(f"done ({elapsed:.1f}s)") 

        # The model occasionally returns more than asked; keep only what we want.
        if len(captions) > num_captions:
            captions = captions[:num_captions]

        print(f"OK ({len(captions)} captions)")

        entry = dict(item) if isinstance(item, dict) else {"scene": scene}  # copy input so metadata/other sources ride along
        entry["scene"] = scene
        entry["name"] = name

        if caption_mode == "keyed":
            entry[caption_key] = captions
        else:
            deterministic = item.get("prompt") if isinstance(item, dict) else None
            legacy_caps = list(captions)
            if deterministic:
                legacy_caps.append(deterministic)  # old-style output folds the deterministic caption in as an extra alternative
            for k in [k for k in entry
                      if k == "caption" or (k.startswith("caption") and k[len("caption"):].isdigit())]:
                del entry[k]  # drop the copied caption fields before rewriting them
            for idx, cap in enumerate(legacy_caps):
                entry["caption" if idx == 0 else f"caption{idx}"] = cap
            entry["caption_model"] = model
            entry["caption_backend"] = backend
        results.append(entry)
        generated += 1

        if generated % 10 == 0:
            _write(output_path, results)

    _write(output_path, results)
    print(
        f"\nDone. Generated {generated} new captions with {backend}/{model}, "
        f"{skipped} resumed, {errors} errors -> {output_path}"
    )

    if with_images:
        print(
            f"Images: {images_used} scene(s) supplemented with a rendered crop, "
            f"{images_missing} had no usable image (sent text-only)."
        )

    print(f"\nReprompts: {total_reprompts} total across {len(reprompted_scenes)} scene(s).")
    if all_bad_chars:
        char_list = ", ".join(
            f"{ch!r} (U+{ord(ch):04X})" for ch in sorted(all_bad_chars)
        )
        print(f"Bad characters encountered: {char_list}")
    else:
        print("Bad characters encountered: none")
    if reprompted_scenes:
        print("Scenes that needed a reprompt:")
        for scene_name, count in reprompted_scenes.items():
            times = "time" if count == 1 else "times"
            print(f"  - {scene_name} (reprompted {count} {times})")

    if prompt_log_file:
        if last_ascii_grid is not None:
            log_text = build_prompt_log_text(
                num_captions, dict_string, last_metadata, grid_label, last_ascii_grid,
                image_clause=image_clause,
            )
        else:
            log_text = build_prompt_log_text(
                num_captions, dict_string, "No metadata available.", grid_label,
                "(no scenes were sent to the LLM this run)", image_clause=image_clause,
            )
        os.makedirs(os.path.dirname(os.path.abspath(prompt_log_file)) or ".", exist_ok=True)
        with open(prompt_log_file, "w", encoding="utf-8") as f:
            f.write(log_text)
        print(f"\nPrompt log written to {prompt_log_file}")
    else:
        print("\n" + "=" * 70)
        if last_full_prompt is not None:
            print("Final full prompt sent (fully rendered: dictionary, metadata, and grid filled in):")
            print("=" * 70)
            print(last_full_prompt)
        else:
            print("Full prompt template used (no scenes were sent to the LLM this run):")
            print("=" * 70)
            print(prompt_template)


def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered captions for MM2 ASCII datasets via Ollama."
    )
    parser.add_argument(
        "--game",
        required=True,
        choices=list(GAME_SETTINGS),
        help=(
            "Which game's tileset settings to use: MM2 (Mario Maker 2), MM (Mega Man), "
            "or MM-Simple (Mega Man simple tileset). Sets the default --tileset/--tileset-we."
        ),
    )
    parser.add_argument("--dataset", required=True, help="Input dataset JSON.")
    parser.add_argument(
        "--tileset",
        default=None,
        help=(
            "Tileset JSON the dataset was built with. Defaults to the tileset for "
            "--game (e.g. MM2 -> mm2_tileset_we.json)."
        ),
    )
    parser.add_argument("--output", required=True, help="Output captioned JSON.")
    parser.add_argument(
        "--backend",
        choices=["ollama", "claude", "openai", "gemini", "smolvlm", "moondream", "llava"],
        default="ollama",
        help=(
            "LLM backend to use. 'smolvlm' runs a local HuggingFace SmolVLM model "
            "in-process via transformers (no API key, no server). 'moondream' and "
            "'llava' are vision models served through the local Ollama server (same "
            "as 'ollama', but they default to those models). Default: ollama"
        ),
    )
    parser.add_argument(
        "--api-key-file",
        default=None,
        help=(
            "Path to a .txt file whose first line is the full API key. "
            "Required when --backend is claude, openai, or gemini."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help=(
            "Max output tokens for the claude/openai/gemini backends "
            "(5 captions need more room than 1). Default: 900"
        ),
    )
    parser.add_argument(
        "--num-captions",
        type=int,
        default=5,
        help=(
            "How many captions to generate per level. The prompt adapts: with 1 it asks "
            "for a single natural caption, with 2+ it asks for that many that vary widely "
            "in length and register. Default: 1"
        ),
    )
    parser.add_argument(
        "--caption-mode",
        choices=["legacy", "keyed"],
        default="legacy",
        help=(
            "Output schema. 'legacy' (default) writes 'caption'/'caption1'/... fields. "
            "'keyed' writes the captions as a list under --caption-key (default "
            "'<model>_captions'), so a scene can carry captions from several models at once. "
            "In both modes every other input attribute (metadata and captions from other "
            "sources) is copied to the output, so passing a previously-captioned dataset back "
            "in accumulates sources rather than replacing them."
        ),
    )
    parser.add_argument(
        "--caption-key",
        default=None,
        help=(
            "Key to store the caption list under when --caption-mode keyed. "
            "Default: '<model>_captions' (e.g. 'qwen2.5:14b_captions')."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. For --backend ollama, default: qwen2.5:14b "
            "(pull with: ollama pull qwen2.5:14b; smaller fallback: qwen2.5:7b or llama3.1:8b). "
            "For --backend claude, default: claude-sonnet-4-6. "
            "For --backend openai, default: gpt-4o. "
            "For --backend gemini, default: gemini-2.5-flash. "
            "For --backend moondream, default: moondream "
            "(pull with: ollama pull moondream). "
            "For --backend llava, default: llava:7b (pull with: ollama pull llava:7b)."
        ),
    )
    parser.add_argument(
        "--url",
        default="http://localhost:11434/api/generate",
        help="Ollama API endpoint. Default: http://localhost:11434/api/generate",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help=(
            "Baseline Ollama context window (tokens). Acts as a floor: each scene's "
            "window is grown from here up to --max-num-ctx when its prompt needs more "
            "room (e.g. the wide token grid), so generation is never starved. Too large "
            "a floor blows the KV cache past VRAM on big models. Default: 8192"
        ),
    )
    parser.add_argument(
        "--max-num-ctx",
        type=int,
        default=16384,
        help=(
            "Ceiling for the per-scene Ollama context window. The window is sized to "
            "fit each prompt plus generation room, capped here. A scene whose prompt "
            "can't fit even at this cap is skipped with a message (use --grid-format "
            "ascii, which is far smaller, or raise this). Raise carefully: big models "
            "(12B+) can blow VRAM at high values. Default: 16384"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help=(
            "Ollama sampling temperature. A small non-zero value avoids the empty "
            "output gemma-family models fall into under greedy (0) decoding. Default: 0.4"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=500,
        help="Per-request timeout in seconds. Default: 120",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
        help="Retry attempts on network failure. Default: 10",
    )
    parser.add_argument(
        "--max-reprompts",
        type=int,
        default=10,
        help=(
            "How many times to re-ask the LLM when its captions contain "
            "non-English characters. When all attempts are exhausted without a "
            "clean response, the level is dropped from the output instead of "
            "crashing the run. Default: 3"
        ),
    )
    parser.add_argument(
        "--grid-format",
        choices=["ascii", "tokens"],
        default="ascii",
        help=(
            "How the level grid is rendered in the prompt. 'tokens' (default) renders "
            "each cell as a 'T<NN>' token numbered per --tileset-we, which is simpler "
            "for LLMs to read and count. 'ascii' uses the raw tile characters from "
            "--tileset, same as the original behavior."
        ),
    )
    parser.add_argument(
        "--tileset-we",
        default=None,
        help=(
            "Tileset JSON defining the T0x token numbering, used only when "
            "--grid-format tokens. Defaults to the tileset for --game."
        ),
    )
    parser.add_argument(
        "--ascii-output-dir",
        default=None,
        help=(
            "Optional folder to dump the exact grid text (ascii or token format, "
            "matching --grid-format) sent to the LLM for each scene, one .txt file per scene."
        ),
    )
    parser.add_argument(
        "--prompt-log",
        default="MM2_Prompt.txt",
        help=(
            "Write the full rendered prompt (system instructions, symbol dictionary, "
            "metadata, and ascii grid for the last scene processed) to this .txt file "
            "at the end of the run. Default: MM2_Prompt.txt"
        ),
    )
    parser.add_argument(
        "--no-prompt-log",
        action="store_true",
        help="Disable writing the prompt log file; print the final prompt to the console instead.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help=(
            "Directory holding each level's rendered PNG crop (the ones produced by "
            "mm2pipeline_data.dataset build --with_images, referenced by the dataset's "
            "'image' field). Defaults to the dataset JSON's own folder. Only matters "
            "when --model is vision-capable (claude, gpt-4o, gemini, or an Ollama "
            "vision model such as llama3.2-vision / llava / qwen2.5vl) -- such models "
            "automatically get each scene's crop sent alongside the ASCII/token grid, "
            "and it is an error if the dataset has no 'image' field to send."
        ),
    )
    args = parser.parse_args()

    # Fill the tileset paths from the selected game unless the user passed them explicitly.
    game_settings = GAME_SETTINGS[args.game]
    if args.tileset is None:
        args.tileset = game_settings["tileset"]
    if args.tileset_we is None:
        args.tileset_we = game_settings["tileset_we"]

    for path, label in [(args.dataset, "dataset"), (args.tileset, "tileset")]:
        if not os.path.isfile(path):
            print(f"Error: {label} not found: {path}")
            sys.exit(1)

    if args.grid_format == "tokens" and not os.path.isfile(args.tileset_we):
        print(f"Error: tileset-we not found: {args.tileset_we}")
        sys.exit(1)

    if args.images_dir and not os.path.isdir(args.images_dir):
        print(f"Error: --images-dir not found: {args.images_dir}")
        sys.exit(1)

    api_key = None
    if args.backend in ("claude", "openai", "gemini"):
        if not args.api_key_file or not os.path.isfile(args.api_key_file):
            print(
                f"Error: --api-key-file (a .txt file with the API key on its first line) "
                f"is required for --backend {args.backend}"
            )
            sys.exit(1)
        api_key = load_api_key(args.api_key_file)

    if args.num_captions < 1:
        print("Error: --num-captions must be at least 1")
        sys.exit(1)

    # The ceiling can't sit below the floor; if it does, lift it to the floor so the
    # per-scene window can at least equal the requested baseline.
    if args.max_num_ctx < args.num_ctx:
        print(
            f"Note: --max-num-ctx ({args.max_num_ctx}) is below --num-ctx ({args.num_ctx}); "
            f"raising it to {args.num_ctx}."
        )
        args.max_num_ctx = args.num_ctx

    default_models = {
        "claude": "claude-sonnet-4-6",
        "openai": "gpt-4o",
        "gemini": "gemini-2.5-flash",
        "ollama": "qwen2.5:14b",
        "smolvlm": "HuggingFaceTB/SmolVLM-Instruct",
        "moondream": "moondream",
        "llava": "llava:7b",
    }
    model = args.model or default_models[args.backend]

    generate_captions(
        args.dataset,
        args.tileset,
        args.output,
        model,
        args.url,
        args.timeout,
        args.retries,
        grid_format=args.grid_format,
        tileset_we_path=args.tileset_we,
        ascii_output_dir=args.ascii_output_dir,
        backend=args.backend,
        api_key=api_key,
        max_tokens=args.max_tokens,
        max_reprompts=args.max_reprompts,
        num_captions=args.num_captions,
        prompt_log_file=None if args.no_prompt_log else args.prompt_log,
        num_ctx=args.num_ctx,
        max_num_ctx=args.max_num_ctx,
        temperature=args.temperature,
        images_dir=args.images_dir,
        caption_mode=args.caption_mode,
        caption_key=args.caption_key,
    )


if __name__ == "__main__":
    main()
