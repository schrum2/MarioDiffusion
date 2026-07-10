"""

NEEDS TO BE MERGED WITH MM2_Files/MarioMaker_llm_captions

This script loads Mega Man levels in VGLC-ASCII format and captions them with an LLM

A loop runs over every scene in the dataset, prompting the LLM with the tileset
key and the ASCII level grid, and assigns each scene a generated caption. The
captions are collected into a list (and optionally written to json) before being
returned.
"""
from pathlib import Path
from collections import Counter
import os
import json
import time
import argparse

import ollama

from create_level_json_data import load_levels
from captions.util import extract_tileset
from util.descriptive_tilesets import GAMES


# for ollama
OLLAMA_NUM_CTX = 32768
EXPECTED_CAPTIONS = 5
MAX_CAPTION_RETRIES = 20 # how many times we tolerate responses that don't contain the correct number of captions


# Default model per --llm branch, used when --model isn't supplied. The openai and claude
# branches hit APIs; ollama runs a local model
DEFAULT_MODELS = {
    "claude": "claude-haiku-4-5",
    "openai": "gpt-5.1",
    "ollama": "qwen3.5:9b",
}


SYSTEM_PROMPT =  """
You are a Mega Man captioning agent; given an ASCII grid representation of a Mega Man level,
an ASCII tile set key, and deterministic level data. you must generate EXACTLY FIVE diverse captions 
that all describe the level accurately. 

RULES:
- Your captions should each DISTINCTLY vary in tone, length, wordiness, playfulness, specificity, etc.
while remaining accurate. Make the diversity noticeable, including short and long captions, playfully
descriptive captions and monotone, serious captions, and so on. E.g., in some captions, include specific
enemy names while in others refer to them as ground enemies/flying enemies, etc.
- Do not mention specific tile types in your answer that you see in the tile set (B, p, etc.),
just describe the level with words.
- Your captions should primarily focus on level structure, and features in the level, typically 
with relative locations, although not explicitly required. Mention specific structures/features like
platforms, enemies, corridors, etc.
- Caption the level like you're writing a prompt to generate it; this means specificity and directness is essential.
- All of the levels you're captioning are roughly square shaped; don't call them "wide" or "tall", as they're square.
This vocabulary could be used for certain structures within the level, like a short horizontal corridor/thin vertical shaft.

ORIENTATION: The first grid row is the TOP of the level and the last row is the BOTTOM; gravity points
down, toward the last row. The player spawn is where the player STARTS, and the player progresses AWAY from it
toward the far end of the level, towards the level exit. Travel toward the top of the grid is ascending (climbing up); 
travel toward the bottom is descending (dropping or falling down). If there is no spawn/exit in the level frame, don't 
guess between ascending and descending; simply classify the scene as vertical (chamber, segment, shaft, etc.)


FORMATTING:
- Your response must contain nothing but the five diverse captions.
- Put each caption on its own line, with no blank lines between them.
- Do not number the captions, add bullets, or write any other text.
- You must write exactly FIVE captions; no more, no less.
- Do not include any dashes or semicolons. The only punctuation you should 
use are commas and periods (, and .). Keep commas rare and only within a single
phrase, and do not chain multiple distinct ideas together with commas. 
- Encapsulate each distinct idea or feature in its own concentrated phrase ended by a
period, rather than stringing many ideas into one run-on sentence. Your captions should still vary
freely in tone, length, and wordiness, never homogeneous in format or structure.

EXAMPLE CAPTIONS:
These are examples of desirable captions that encapsulate ideas/level features into discrete '.'-separated chunks
while still varying in tone, specificity, length, etc.:
-  Multiple vertical passages interweave through this towering shaft. Snipers guard the lower levels. 
Fire pillars erupt periodically. The exit waits high above.
- An extensive horizontal descent beginning from a modest platform on the left side. The player travels rightward 
across progressively lower terrain featuring moving platforms and scattered enemies including Bunby Helis and a Sniper Joe. 
Multiple weapon power-ups dot the landscape while deadly spikes appear in the lower sections. The exit awaits far to the right at the bottom level.
- A claustrophobic descent begins here. One wall-crawler blocks the passage near the start. Further down, the area 
opens into a gauntlet featuring ranged enemies, moving platforms, and eventually a water-filled cavern where bouncing enemies reside.

REMINDERS:
- Make sure your captions are each NOTICEABLY DISTINCT from one another in length, tone, and specificity:
- For length: at least one caption should be long, one should be short, and the rest should
fall in between.
- For specificity: one or two captions should contain specific enemy/power-up names, while others can be vague and state enemy class (ranged, ground).
- You must return FIVE distinct captions, each on their own line.
- Don't say things like "This level has..." or "The level features...". Just directly
describe the level itself without mentioning "level".
"""



# Trailing reminder appended as the final (freshest) user msg 
CAPTION_REMINDER = ("Reminder: output exactly five captions and nothing else, each on its own line with no numbering, labels, or blank lines. "
"Break each caption into short '.'-separated phrases, and make the five intentionally diverse in length, specificity, and tone. "
"For vertical segments, do not guess at whether the segment is ascending or descending, unless it is absolutely clear by the level structure/metadata "
"that you observe and are provided. For ambiguous vertical scenes, note the structure and verticality but don't assume directionality. "
"You may still classify structures within a scene as ascending or descending, especially if there are notable structures "
"present in horizontal (always left-to-right) level scenes. Don't explicitly state that a level is left-to-right; that information is already known."
)



def scene_to_ASCII(scene: list[list[int]], id_to_char: dict[int, str],
                   null_ids: frozenset[int] = frozenset()) -> list[str]:
    """
    Decode a 2D integer tile-id grid into a list of ASCII row strings using the tileset map.

    This is used only to build the captioning prompt; the integer grid itself is carried
    through to the output unchanged.

    Drop the leading all-null padding rows and render any
    remaining null cell as empty space ('-', the air char) so the LLM never sees the '@' void
    and invents a feature ("void", out of bounds, etc.) from it. This matches how
    deterministic_caption treats null tiles (open/non-terrain), keeping the grid the LLM reads
    consistent with the metadata it's grounded on
    """
    first_real = next((r for r in range(len(scene)) if not all(t in null_ids for t in scene[r])), 0)
    return ["".join("-" if tile in null_ids else id_to_char[tile] for tile in row)
            for row in scene[first_real:]]


def load_dataset(path: str, char_to_id: dict[str, int]) -> list[tuple[list[list[int]], str, dict]]:
    """
    Load integer tile-id scenes for an LLM to caption.

    {path} could be:
        - a JSON file produced by create_megaman_json_data.py (or a previous captioning pass):
          a list of entries, each a dict with a "sample" key holding the 2D integer tile-id grid
          (a "scene" key or a bare grid are also accepted). Any other keys the entry carries --
          metadata (source_level, scan_mode, data, ...) and captions from earlier sources
          (deterministic_captions, some_model_captions, ...) -- are returned as `attrs` so main()
          can copy them straight through to the output. This is what lets a captioned dataset be
          fed back in to accumulate sources instead of replacing them.

        - a directory of VGLC-ASCII level files (*.txt), one whole level per file. These
          are encoded to integer grids (via char_to_id) so the output format is the same
          either way. There is no metadata to carry, so `attrs` is empty.

    Returns a list of (scene, label, attrs) tuples, where each scene is a 2D grid of integer
    tile ids, label identifies the source, and attrs is the dict of all other input attributes
    (empty for directory input). The integer grid is what gets stored in the output; it is only
    decoded to ASCII transiently for the captioning prompt.
    """
    # A directory of level files: load_levels reads every *.txt, strips blank
    # lines/trailing whitespace, and returns one list-of-rows per file. Encode each to an
    # integer grid so the rest of the pipeline is uniformly integer tile ids. Pair each
    # scene with its source filename; load_levels and this glob share the sorted("*.txt") order.
    if os.path.isdir(path):
        levels = load_levels(path)
        files = sorted(Path(path).glob("*.txt"))
        return [([[char_to_id[c] for c in row] for row in level], file.name, {})
                for level, file in zip(levels, files)]

    # Otherwise treat it as a JSON file of integer tile-id scenes.
    with open(path, "r") as f:
        data = json.load(f)

    scenes = []
    for entry in data:
        if isinstance(entry, dict):
            # create_megaman_json_data.py writes the grid under "sample"; accept "scene" too.
            grid = entry["sample"] if "sample" in entry else entry["scene"]
            # Everything except the grid field itself is metadata/other captions to carry through.
            attrs = {k: v for k, v in entry.items() if k not in ("sample", "scene")}
        else:
            grid = entry
            attrs = {}
        scenes.append((grid, Path(path).name, attrs))
    return scenes


def filter_tile_set(scene: str, tileset: dict) -> dict:
    """
    Given an ASCII level scene and the complete tile set for the game the given scene belongs to, return a filtered
    tile set dict to insert in the LLM prompt to convserve (a marginal amount of) tokens, and to avoid hallucination/confusion
    in the LLM response. This filtered tileset only contains the k: v pairs that are found in the provided scene.

    Args:
        scene (str):  the ASCII level scene
        tileset(dict -- char: str): the complete ASCII char: string description tile set
    
    Returns:
        dict -- char: str: the filtered tile set that only contains tiles found in the level
    """

    filtered = {char: desc for char, desc in tileset.items() if char in scene} # this function could literally be a one-liner I'm not sure why I decided to make this an entire function I'll probably remove this later
    return filtered


# Tile chars that carry no object information and are left out of the metadata object
# counts. Ground/wall ("#") is excluded too because it is summarized by the terrain
# column heights instead of being counted cell-by-cell.
_METADATA_SKIP_CHARS = {"P", "-", "t", "@", "#"}


def deterministic_caption(scene: list[list[int]], id_to_char: dict[int, str], char_to_id: dict[str, int],
                          tile_descriptors: dict, names: dict, describe_locations: bool = False,
                          describe_absence: bool = False, data: dict = None) -> str:
    """
    Build a block of pre-computed structural metadata for an integer tile-id scene, to feed
    the LLM as grounding context (replacing the old MM_create_ascii_captions caption).

    The metadata is purely mechanical fact about the grid that the LLM can lean on instead of
    re-deriving it from the raw ASCII:
      - raw occupied-cell counts per object tile type (readable names)
      - terrain top-of-column heights (structural solid tiles, ignoring enemies/hazards)
      - floor / ceiling analysis (top and bottom rows)
      - left / center / right region column boundaries

    Args:
        scene: 2D grid of integer tile ids (the raw scene, not the ASCII decode).
        id_to_char, char_to_id, tile_descriptors: tileset maps from extract_tileset.
        names: the game's descriptive tileset dict (char -> readable description) used to render
            the object-count lines with human names.
        describe_locations / describe_absence / data: kept for signature compatibility with the
            caller; this metadata builder doesn't use them.

    Returns:
        A multi-line metadata string.
    """

    def is_null(tile: int) -> bool:
        return "null" in tile_descriptors.get(id_to_char.get(tile), set())

    # Scenes are padded up to 16x16 for the diffusion UNet, which adds rows of all-null
    # tiles above the real 16x14 level. Drop those leading null rows so the ceiling analysis
    # and terrain heights describe the actual playable area rather than the padding.
    first_real = next((r for r in range(len(scene)) if not all(is_null(t) for t in scene[r])), len(scene))
    scene = scene[first_real:] or scene

    height = len(scene)
    width = len(scene[0])

    # Terrain = structural solid tiles (ground, walls, blocks); enemies and hazards that
    # happen to be "solid" don't count as terrain.
    def is_terrain(tile: int) -> bool:
        desc = tile_descriptors.get(id_to_char.get(tile), set())
        return "solid" in desc and "enemy" not in desc and "hazard" not in desc

    # Object tile counts: raw occupied cells per type, by readable name, most common first 
    counts = Counter()
    for row in scene:
        for tile in row:
            char = id_to_char.get(tile)
            if char is None or char in _METADATA_SKIP_CHARS:
                continue
            counts[char] += 1
    if counts:
        # Some tileset descriptions carry a ":" gloss (e.g. "Hazard Blocks: extends ..."); keep
        # just the name before it so the count line reads cleanly ("Hazard Blocks: 2").
        count_lines = "\n".join(
            f"  {names.get(char, char).split(':')[0].strip()}: {n}"
            for char, n in sorted(counts.items(), key=lambda kv: -kv[1])
        )
    else:
        count_lines = "  (none)"

    # Per-column ground profile: the elevation of the LOWEST standable floor surface, found by
    # scanning up from the bottom. "Standable" reuses the astar/MegaManState (addOrb/placeSpawn) rule:
    # a solid tile is a surface only if the two cells above it are open, since Mega Man is
    # two tiles tall. Floating platforms higher up are ignored (the lowest surface
    # wins); the model reads the grid for those. A column with no standable surface is reported as
    # "wall" (mostly solid, blocked) or "pit" (no safe footing: open drop or hazard-sealed floor).
    def is_open(tile: int) -> bool:
        return "solid" not in tile_descriptors.get(id_to_char.get(tile), set())

    ground = []
    for c in range(width):
        # Walk the column from the bottom row upward (stopping at row 2 so the two cells of head
        # clearance above always exist) and take the first solid tile that has open space above it.
        surface = next(
            (r for r in range(height - 1, 1, -1)
             if is_terrain(scene[r][c]) and is_open(scene[r - 1][c]) and is_open(scene[r - 2][c])),
            None,
        )
        if surface is not None:
            # Elevation = tiles between that surface and the bottom edge (0 = floor on the bottom row).
            ground.append(str((height - 1) - surface))
        else:
            # Nowhere to stand (open, or floor sealed by hazards): wall if mostly solid, else pit.
            solid = sum(1 for r in range(height) if is_terrain(scene[r][c]))
            ground.append("wall" if solid * 2 >= height else "pit")
    # Right-pad each token to a fixed width so the per-column values line up in one readable row.
    ground_line = " ".join(f"{tok:>4}" for tok in ground)

    # Ceiling (top row): solid coverage and contiguous gap count. The ground profile already
    # describes the floor, but nothing else covers overhead terrain, so the top row is summarized
    # here. It only counts as a real ceiling when solid tiles cover most of the row; a few solid
    # cells up top are just the tips of structures poking through, not a ceiling.
    def edge_summary(row: list[int]) -> str:
        solid = sum(1 for t in row if is_terrain(t))
        if solid == 0:
            return "absent"
        if solid == width:
            return "solid across"
        if solid * 2 < width:
            return "mostly open"
        gaps, in_gap = 0, False
        for t in row:
            if is_terrain(t):
                in_gap = False
            elif not in_gap:
                gaps += 1
                in_gap = True
        return f"present with {gaps} gap" + ("s" if gaps != 1 else "")

    ceiling = edge_summary(scene[0])

    # Region boundaries: split the width into left / center / right thirds 
    left_end = width // 3
    center_end = 2 * width // 3
    regions = (f"left=cols 1-{left_end}, center=cols {left_end + 1}-{center_end}, "
               f"right=cols {center_end + 1}-{width}")

    return (
        "Object tile counts (raw occupied cells per type, one placed object may span several cells):\n"
        f"{count_lines}\n\n"
        f"Ground surface profile per column (columns 1-{width} left to right; each value is how "
        "many tiles the lowest standable floor sits above the bottom, 0=floor at the very bottom; "
        "'wall'=solid blocked column, 'pit'=no safe footing (an open drop, or a floor sealed off "
        "by hazards)):\n"
        f"  {ground_line}\n"
        f"Ceiling (top row): {ceiling}\n\n"
        "Region boundaries (use these when assigning left/center/right): "
        f"{regions}"
    )



def llm_caption(scene: str,  deterministic: str, game: str = "Mega Man", tileset: dict = None, llm: str = "ollama", model: str = "qwen3.5:9b") -> list[str]:

    
    deterministic_msg = (
        "For grounding and reference, here is pre-computed structural metadata for this level. Treat it as "
        "accurate, stay consistent with it, and don't invent features it doesn't support or "
        "re-count terrain from the grid yourself. How to read it:\n"
        "- Object tile counts are RAW occupied-cell counts, NOT object counts. One placed thing "
        "(a ladder, a stretch of water, a block structure) can span many cells, so a large count "
        "usually means one big feature, not many separate ones. Use rough quantities, never exact "
        "tile numbers.\n"
        "- Ground surface profile per column gives the level's walkable floor shape: each value is "
        "how high the lowest standable floor sits in that column (rising values = steps/hills, a "
        "flat run = flat ground), while 'wall' marks a solid blocked column and 'pit' marks a column "
        "with no safe footing (an open drop, or a floor sealed off by hazards). It only tracks the "
        "lowest floor, so read the grid for raised platforms or overhead structures above it. This output " 
        "should shape your classification of the ground, not simply the tiles at the very bottom row. "
        "Keep in mind that the player always moves left-to-right in non-vertical segments, so base "
        "your ascending vs descending structure analysis based on this. DO NOT explicitly mention 'ground levels' "
        "in your captions; use the data to inform your natural captions.\n"
        "- Ceiling describes the top row's overhead terrain; region boundaries map columns to "
        "left/center/right.\n"
        "Still write the captions in your own words per the rules above:\n"
        f"{deterministic}"
    )
        
    # only load dotenv module if we need an api key (claude/openai)
    if llm != "ollama":
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent / '.env'
        load_dotenv(env_path)

   

    # claude branch
    if llm == "claude":
        """
        Prompt claude (via API) w/ ASCII level scene and tileset, return the caption(s) it generates
        """

        from anthropic import Anthropic
        ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        client = Anthropic(api_key=ANTHROPIC_API_KEY)


        tileset_str = json.dumps(tileset, indent=2)

        context = [
            {"role": "user", "content": f"Here is the tile set for {game}:\n{tileset_str}"},
            {"role": "user", "content": f"Level Scene:\n{scene}"},
            {"role": "user", "content": deterministic_msg},
            {"role": "user", "content": CAPTION_REMINDER},
        ]

        # sup claude
        message = client.messages.create(
                    max_tokens=2048,
                    system=SYSTEM_PROMPT, # claude requires system prompt to be separated from context block
                    messages=context,
                    model=model
                    )
        
        # message.content is a list of content blocks; pull the text out and split into a list
        # separated by line breaks, dropping blank lines (Claude often separates captions with blank lines)
        captions = [line.strip() for line in message.content[0].text.split("\n") if line.strip()]

        # print(f"[{len(captions)} captions detected]\n")
        return captions
    


    # openai branch
    elif llm == "openai":
        """
        Prompt openai (via API) w/ ASCII level scene and tileset, return the caption(s) it generates
        """
        from openai import OpenAI
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        client2 = OpenAI(api_key= OPENAI_API_KEY)



        tileset_str = json.dumps(tileset, indent=2)

        context = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the tile set for {game}:\n{tileset_str}"},
            {"role": "user", "content": f"Level Scene:\n{scene}"},
            {"role": "user", "content": deterministic_msg},
            {"role": "user", "content": CAPTION_REMINDER},
        ]

        # sup mr. altman
        completion = client2.chat.completions.create(
        model=model,
        messages=context,
        )
        message = completion.choices[0].message.content
        
        captions = [line.strip() for line in message.split("\n") if line.strip()]

        # print(f"[{len(captions)} captions detected]\n")
        return captions

    # local branch
    elif llm == "ollama":
        """
        Prompt a local ollama model once per style in LOCAL_SYSTEM_PROMPT_STYLES, each call asking
        for a single caption in that style, and return the collected list. Splitting the five
        captions into five focused single-caption calls keeps the load on a small model light and
        makes the per-style diversity deterministic instead of something the model has to invent.
        """
        tileset_str = json.dumps(tileset, indent=2)

    
        context = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the tile set for {game}:\n{tileset_str}"},
            {"role": "user", "content": f"Level Scene:\n{scene}"},
            {"role": "user", "content": deterministic_msg},
            {"role": "user", "content": CAPTION_REMINDER}
        ]

      
        # Retry until we get a compliant response, surfacing each failed attempt, and give up after MAX_CAPTION_RETRIES.
        captions = []
        for attempt in range(1, MAX_CAPTION_RETRIES + 1):
            completion = ollama.chat(
                model=model,
                messages=context,
                think=False,
                options={"num_ctx": OLLAMA_NUM_CTX, "temperature": 0.4},
            )
            message = completion.message.content

            captions = [line.strip() for line in message.split("\n") if line.strip()]

            if len(captions) == EXPECTED_CAPTIONS:
                return captions

            print(f"[ollama retry] Attempt {attempt}/{MAX_CAPTION_RETRIES} returned "
                  f"{len(captions)} caption(s), expected {EXPECTED_CAPTIONS}; retrying...\n")

        print(f"[ollama] Gave up after {MAX_CAPTION_RETRIES} attempts; last response had "
              f"{len(captions)} caption(s) instead of {EXPECTED_CAPTIONS}.\n")
        return captions

    else:
        print("You've provided an invalid LLM inference mode: Please try again and select one of the following: claude, openai, ollama ")



def parse_args():
    argparser = argparse.ArgumentParser(
        description="Caption VGLC-ASCII Mega Man levels with a local LLM."
    )

    argparser.add_argument("--levels", default="../TheVGLC/MegaMan/Enhanced",
                           help="JSON dataset of integer tile-id scenes from create_megaman_json_data.py, "
                                "or a directory of VGLC-ASCII level .txt files")
    argparser.add_argument("--game", default="megaman", choices=list(GAMES),
                           help="Which Mega Man game/tileset to caption for. Selects the descriptive "
                                "tileset (from util/descriptive_tilesets.py), the prompt game name, and "
                                "the tileset JSON used to decode integer scenes to ASCII")
    argparser.add_argument("--llm", choices=["claude", "openai", "ollama"], default="ollama",
                           help="The source of the LLM inference used to caption the provided level scenes. The openai and claude choices use APIs, while ollama runs a local model")
    argparser.add_argument("--model", default=None,
                           help="Model to prompt for captions. When omitted, defaults per --llm branch: "
                                "claude -> claude-haiku-4-5, openai -> gpt-5.1, ollama -> qwen3.5:9b")
    argparser.add_argument("--output", default=None,
                           help="Optional path to write the captioned [{scene, caption}] list as JSON")
    argparser.add_argument("--limit", type=int, default=None,
                           help="Max number of scenes to caption. Defaults to the entire dataset")
    argparser.add_argument("--caption-mode", default="keyed", choices=["legacy", "keyed"],
                           help="Output schema. 'keyed' (default) writes the captions as a list under --caption-key (default "
                                "'<model>_captions'), so a scene can carry captions from several models at once. 'legacy' writes "
                                "'caption'/'caption1'/... fields instead. In both modes every other input attribute (metadata and captions from other "
                                "sources) is copied to the output, so passing a previously-captioned dataset back in accumulates sources rather than replacing them.")
    argparser.add_argument("--caption-key", default=None,
                           help="Key to store the caption list under when --caption-mode keyed. Defaults to '<model>_captions' "
                                "(the resolved model name), so each model's captions land under their own key.")


    return argparser.parse_args()



def main() -> list[list[str]]:

    args = parse_args()

    # Resolve the selected game: its human-readable name (for the prompt), its descriptive
    # tileset dict (char -> readable description, for the prompt key and object-count names),
    # and its tileset JSON (structural descriptors + id<->char maps for decoding scenes).
    game = GAMES[args.game]
    game_name = game["name"]
    tile_names = game["tiles"]["tiles"]
    tileset_path = game["tileset"]

    # Tileset map: integer id -> ASCII char, used to decode each integer scene into the
    # ASCII grid shown to the LLM. char -> id is used to encode a directory of ASCII levels.
    # tile_descriptors drives the deterministic structural caption fed to the LLM as context.
    _, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)

    # Tile ids flagged "null" in the tileset are out-of-bounds padding / void, not level
    # content; scene_to_ASCII uses this to strip the padding rows and hide the '@' void
    # from the LLM so it doesn't describe a nonexistent feature.
    null_ids = frozenset(tid for tid, ch in id_to_char.items() if "null" in tile_descriptors.get(ch, set()))

    # load integer tile-id scenes
    scenes = load_dataset(args.levels, char_to_id)


    # parsers all of scenes when limit is None 
    scenes = scenes[:args.limit]


    caption_lists = [] # list[list] of caption set for each scene
    captioned_dataset = []

    # Resolve the model: --model overrides, otherwise fall back to the per-branch default.
    model = args.model or DEFAULT_MODELS[args.llm]

    # Key the keyed-mode caption list under --caption-key, or '<model>_captions' by default so
    # each model's captions accumulate under their own field when datasets are chained.
    caption_key = args.caption_key or f"{model}_captions"

    # Label combining the inference source and resolved model; printed and stored per entry.
    llmstr = f"{args.llm} - {model}"

    # Mark the start of the full captioning process to report total elapsed time at the end.
    start_time = time.time()
    print(f"[timing] Captioning started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")

    # caption each scene, append back to running lists
    for i, (scene, label, attrs) in enumerate(scenes):

        # Decode the integer grid to ASCII rows purely to build the prompt; the integer
        # grid itself is what gets stored back in the output unchanged. Null padding rows
        # are stripped and any out-of-bounds null cells are hidden (rendered as air).
        scene_str = "\n".join(scene_to_ASCII(scene, id_to_char, null_ids))

        # Get filtered tileset for current scene
        filtered_tiles = filter_tile_set(scene_str, tile_names)

        # Fetch the deterministic structural caption for this scene to ground the LLM as
        # an extra context message. Operates on the integer grid, not the ASCII decode.
        det_caption = deterministic_caption(scene, id_to_char, char_to_id, tile_descriptors, names=tile_names)

        # assign and collect captions
        caption_set = llm_caption(scene_str, game=game_name, model=model, tileset=filtered_tiles, llm=args.llm, deterministic=det_caption)
        

        print(f"------------------ [{llmstr}]  [{label}] ({i + 1}/{len(scenes)}) ------------------\n")
        for j, caption in enumerate(caption_set):
            print(f"[Caption {j + 1}/{len(caption_set)}] {caption}\n")

       

        if len(caption_set) != EXPECTED_CAPTIONS:
            print(f"[skip] {label}: got {len(caption_set)} caption(s) instead of {EXPECTED_CAPTIONS}; skipping this scene.\n")
            continue

        caption_lists.append(caption_set)
        # Copy all carried-through input attributes (metadata + captions from earlier sources)
        # first, then (re)write the scene and this run's captions on top, so chaining datasets
        # accumulates sources instead of replacing them. "scene" holds the integer tile-id grid.
        entry = dict(attrs)
        entry["scene"] = scene
        if args.caption_mode == "legacy":
            entry.update({"caption": caption_set[0], "caption1": caption_set[1], "caption2": caption_set[2],
                          "caption3": caption_set[3], "caption4": caption_set[4], "model": llmstr})
        else:
            entry[caption_key] = caption_set
        captioned_dataset.append(entry)

    # Mark the end of the full captioning process and report total elapsed time.
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[timing] Captioning finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
    print(f"[timing] Total captioning time: {elapsed:.2f}s ({elapsed / 60:.2f} min)\n")

    # save to specified output dir if specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(captioned_dataset, f, indent=2)
        print(f"Captioned dataset saved to {args.output}")

    return caption_lists


if __name__ == "__main__":
    main()
