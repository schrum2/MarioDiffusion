"""
This script loads Mega Man levels in VGLC-ASCII format and captions them with an LLM

A loop runs over every scene in the dataset, prompting the LLM with the tileset
key and the ASCII level grid, and assigns each scene a generated caption. The
captions are collected into a list (and optionally written to disk) before being
returned.
"""

import os
import json
import argparse

import ollama

from dotenv import load_dotenv
from pathlib import Path
from anthropic import Anthropic

env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(env_path)

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
client = Anthropic(api_key=ANTHROPIC_API_KEY)


from openai import OpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client2 = OpenAI(api_key= OPENAI_API_KEY)



from create_level_json_data import load_levels
from captions.util import extract_tileset

MM_TILESET_DICT = {
    "tiles" : {
        "P": "Mega Man's starting spawn point",
        "Z": "Level exit point/final goal",
        "@": "Out of bounds, inaccessible null space",
        "-": "Empty space",
        "~": "Water (slows movement)",
        "#": "Solid blocks representing ground or walls",
        "|": "Climbable ladders",
        "B": "Solid but breakable blocks",
        "L": "Large Life Energy power-up",  
        "H": "Deadly solid hazard",
        "q": "Jumping Kamadoma enemy",
        "o": "Flying Mambu enemy",
        "j": "Flying Bunby Heli enemy",
        "c": "Ranged Blaster enemy",
        "^": "Vertical Adhering Suzy enemy",
        "<": "Horizontal Adhering Suzy enemy",
        "f": "Jumping Big Eye enemy",
        "t": "Secret transparent blocks (looks like regular blocks, but Mega Man can phase through them)",
        "A": "Dissapearing/Reappearing blocks (fades in and out)",
        "M": "Moving Platform blocks",
        "D": "Passable Door blocks",
        "W": "Large Weapon Energy power-up",
        "w": "Small Weapon Energy power-up",
        "l": "Small Life Energy power-up",
        "+": "Collectible 1-UP Extra Life Power-up",
        "*": "Collectible Yashichi Power-up",
        "U": "Collectible Magnet Beam Power-up",
        "C": "Elec Block, creates a temporary lighting barrier that extends outward",
        "p": "Ranged Foot Holder enemy, Mega Man can stand and jump from these",
        "r": "Ranged, shielded Sniper Joe enemy",
        "k": "Killer Bomb enemy",
        "g": "Ground Gabyoall enemy",
        "e": "Ranged Screw Driver turret enemy",
        "m": "Jumping exploding Bombombomb enemy",
        "i": "Floating, ranged Watcher enemy",
        "b": "Flying Bunby Heli enemy",
        "a": "Stationary, ranged Met enemy",
        "d": "Ranged Pickelman enemy",
        "h": "Crazy Razy enemy",
        "n": "Flying PePe penguin enemy",
        "I": "Changkey vertical fire pillar enemy"
    }
}


SYSTEM_PROMPT =  """
                    You are a Mega Man captioning agent; given an ASCII grid representation of a Mega Man level
                    and an ASCII tile set key to go along with it, you must generate EXACTLY FIVE diverse captions 
                    that all describe the level accurately. 

                    RULES:
                    - Your captions should each DISTINCTLY vary in tone, length, wordiness, playfulness, specificity, etc.
                    while remaining accurate. Make the diversity noticeable, including short and longer captions, playfully
                    descriptive captions and monotone, serious captions, and so on. Keep the longest captions within 3-4 sentences,
                    never overly long. Your shortest captions should be succinct statements about level features/structure.
                    - Do not mention specific tile types in your answer that you see in the tile set (B, p, etc.),
                    just describe the level with words.
                    - Your captions should primarily focus on level structure, and features in the level, typically 
                    with relative locations, although not explicitly required. Mention specific structures/features like
                    platforms, enemies, corridors, etc.
                    - Caption the level like you're writing a prompt to generate it; this means specificity and directness is essential.
                    
                    ORIENTATION: The first grid row is the TOP of the level and the last row is the BOTTOM; gravity points
                    down, toward the last row. The player spawn is where the player STARTS, and the player progresses AWAY from it
                    toward the far end of the level. Decide ascending vs descending from the player's direction of travel away from
                    the spawn, never from connectivity alone. Travel toward the top of the grid is ascending (climbing up); travel
                    toward the bottom is descending (dropping or falling down). For a tall vertical level, a spawn near the bottom
                    means the level is an ASCENT (the player climbs upward), while a spawn near the top means a DESCENT. Never
                    describe the spawn as somewhere the player descends to or arrives at, it is where they begin. Wide horizontal
                    levels flow from the spawn on the left toward the right.

                    READING SLOPES AND STAIRS: To tell which way a staircase or sloped floor goes, do not eyeball the overall
                    shape and do not judge it by where the bulk of the solid tiles sit. Instead trace the walkable surface,
                    the topmost solid tile the player can stand on, one column at a time across the direction of travel.
                    Because the first row is the top, a surface nearer the top of the grid is physically HIGHER and a surface
                    nearer the bottom is LOWER. If that surface moves UP the grid (toward row one) as the player advances, the
                    stairs ASCEND in that direction; if it moves DOWN the grid (toward the last row), they DESCEND. A block of
                    solid tiles heaped in a bottom corner is almost always steps the player climbs UP toward that corner, not a
                    drop. Only call something a pit, gap, or hole when there is genuinely open space with no solid floor beneath
                    it within a jump, check the rows directly below before claiming a pit, and never invent one.

                    FORMATTING:
                    - Your response must contain nothing but the five diverse captions.
                    - Put each caption on its own line, with no blank lines between them.
                    - Do not number the captions, add bullets, or write any other text.
                    - You must write exactly FIVE captions; no more, no less.
                    - Do not include any dashes or semicolons. The only punctuation you should 
                    use are commas and periods (, and .)
                    - Don't say things like "This level has..." or "The level features...". Just directly
                    describe the level itself without mentioning "level". 
                 """

def scene_to_ASCII(scene: list[list[int]], id_to_char: dict[int, str]) -> list[str]:
    """
    Decode a 2D integer tile-id grid into a list of ASCII row strings using the tileset map.

    This is used only to build the captioning prompt; the integer grid itself is carried
    through to the output unchanged.
    """
    return ["".join(id_to_char[tile] for tile in row) for row in scene]


def load_dataset(path: str, char_to_id: dict[str, int]) -> list[tuple[list[list[int]], str]]:
    """
    Load integer tile-id scenes for an LLM to caption.

    {path} could be:
        - a JSON file produced by create_megaman_json_data.py: a list of entries, each a
          dict with a "sample" key holding the 2D integer tile-id grid (a "scene" key or a
          bare grid are also accepted).

        - a directory of VGLC-ASCII level files (*.txt), one whole level per file. These
          are encoded to integer grids (via char_to_id) so the output format is the same
          either way.

    Returns a list of (scene, label) tuples, where each scene is a 2D grid of integer
    tile ids and label identifies the source. The integer grid is what gets stored in the
    output; it is only decoded to ASCII transiently for the captioning prompt.
    """
    # A directory of level files: load_levels reads every *.txt, strips blank
    # lines/trailing whitespace, and returns one list-of-rows per file. Encode each to an
    # integer grid so the rest of the pipeline is uniformly integer tile ids. Pair each
    # scene with its source filename; load_levels and this glob share the sorted("*.txt") order.
    if os.path.isdir(path):
        levels = load_levels(path)
        files = sorted(Path(path).glob("*.txt"))
        return [([[char_to_id[c] for c in row] for row in level], file.name)
                for level, file in zip(levels, files)]

    # Otherwise treat it as a JSON file of integer tile-id scenes.
    with open(path, "r") as f:
        data = json.load(f)

    scenes = []
    for entry in data:
        if isinstance(entry, dict):
            # create_megaman_json_data.py writes the grid under "sample"; accept "scene" too.
            grid = entry["sample"] if "sample" in entry else entry["scene"]
        else:
            grid = entry
        scenes.append((grid, Path(path).name))
    return scenes


# TODO: Create ONE LLM caption function and make claude/chat/ollama a model parameter 
def chatGPT_caption(scene: str, game: str = "Mega Man", tileset: dict = MM_TILESET_DICT, model: str = "qwen3.5:9b") -> list[str]:
    """
    Prompt claude (via API) w/ ASCII level scene and tileset, return the caption(s) it generates
    """
    tileset_str = json.dumps(tileset, indent=2)

    context = [
        {"role": "user", "content": f"Here is the tile set for {game}:\n{tileset_str}"},
        {"role": "user", "content": f"Level Scene:\n{scene}"},
    ]

    # sup claude
    completion = client2.chat.completions.create(
      model="gpt-5.1",
      messages=context,
    )
    message = completion.choices[0].message.content
    
    captions = [line.strip() for line in message.split("\n") if line.strip()]

    print(f"[{len(captions)} captions detected]\n")
    return captions




def claude_caption(scene: str, game: str = "Mega Man", tileset: dict = MM_TILESET_DICT, model: str = "qwen3.5:9b") -> list[str]:
    """
    Prompt claude (via API) w/ ASCII level scene and tileset, return the caption(s) it generates
    """
    tileset_str = json.dumps(tileset, indent=2)

    context = [
        {"role": "user", "content": f"Here is the tile set for {game}:\n{tileset_str}"},
        {"role": "user", "content": f"Level Scene:\n{scene}"},
    ]

    # sup claude
    message = client.messages.create(
                max_tokens=1024,
                system=SYSTEM_PROMPT, # claude requires system prompt to be separated from context block
                messages=context,
                model="claude-haiku-4-5"
                )
    
    # message.content is a list of content blocks; pull the text out and split into a list
    # separated by line breaks, dropping blank lines (Claude often separates captions with blank lines)
    captions = [line.strip() for line in message.content[0].text.split("\n") if line.strip()]

    print(f"[{len(captions)} captions detected]\n")
    return captions

def llama_caption(scene: str, game: str = "Mega Man", tileset: dict = MM_TILESET_DICT, model: str = "qwen3.5:9b") -> str:
    """
    Prompt a local ollama model with the tileset key and an ASCII level grid,
    and return the generated caption.
    """
    tileset_str = json.dumps(tileset, indent=2)

    context = [
        {"role": "system", "content":
            "Given a tile set key and an ASCII level grid, "
            "generate a descriptive yet succinct caption for the level."},
        {"role": "user", "content": f"Here is the tile set for {game}:\n{tileset_str}"},
        {"role": "user", "content": f"Level Scene:\n{scene}"},
    ]

    completion = ollama.chat(model=model, messages=context)
    caption = completion.message.content
    return caption


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Caption VGLC-ASCII Mega Man levels with a local LLM."
    )

    argparser.add_argument("--levels", default="../TheVGLC/MegaMan/Enhanced",
                           help="JSON dataset of integer tile-id scenes from create_megaman_json_data.py, "
                                "or a directory of VGLC-ASCII level .txt files")
    argparser.add_argument("--tileset", default="datasets/MM.json",
                           help="Tileset JSON used to decode integer scenes to ASCII for the prompt "
                                "(must match the tileset the scenes were generated from)")
    argparser.add_argument("--game", default="Mega Man",
                           help="Game name passed to the LLM prompt for context")
    argparser.add_argument("--model", default="qwen3.5:9b",
                           help="Local ollama model to prompt for captions")
    argparser.add_argument("--output", default=None,
                           help="Optional path to write the captioned [{scene, caption}] list as JSON")
    argparser.add_argument("--limit", type=int, default=None,
                           help="Max number of scenes to caption. Defaults to the entire dataset")
    return argparser.parse_args()


def main() -> list[list[str]]:

    args = parse_args()

    # Tileset map: integer id -> ASCII char, used to decode each integer scene into the
    # ASCII grid shown to the LLM. char -> id is used to encode a directory of ASCII levels.
    _, id_to_char, char_to_id, _ = extract_tileset(args.tileset)

    # load integer tile-id scenes
    scenes = load_dataset(args.levels, char_to_id)

    
    # parsers all of scenes when limit is None 
    scenes = scenes[:args.limit]


    caption_lists = [] # list[list] of caption set for each scene
    captioned_dataset = []

    # caption each scene, append back to running lists
    for i, (scene, label) in enumerate(scenes):

        # Decode the integer grid to ASCII rows purely to build the prompt; the integer
        # grid itself is what gets stored back in the output.
        scene_str = "\n".join(scene_to_ASCII(scene, id_to_char))
        # currently wired to the claude API version, can also be set to local
        caption_set = chatGPT_caption(scene_str, game=args.game, model=args.model)

        
        print(f"------------------[{label}] ({i + 1}/{len(scenes)})------------------\n")
        for j, caption in enumerate(caption_set):
            print(f"[Caption {j + 1}/{len(caption_set)}] {caption}\n")


        caption_lists.append(caption_set)
        # ugly but necessary; want single json object with flat fields scene, cap, cap1, ..., cap4.
        # "scene" holds the original integer tile-id grid, carried through unchanged.
        captioned_dataset.append({"scene": scene, "caption": caption_set[0], "caption1": caption_set[1], "caption2": caption_set[2], "caption3": caption_set[3], "caption4": caption_set[4]})

    # save to specified output dir if specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(captioned_dataset, f, indent=2)
        print(f"Captioned dataset saved to {args.output}")

    return caption_lists


if __name__ == "__main__":
    main()
