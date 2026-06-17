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

from create_level_json_data import load_levels

MM_TILESET_DICT = {
    "tiles" : {
        "P": ["passable", "empty", "spawn"],
        "@": ["null"],
        "-": ["passable", "empty"],
        "~": ["passable", "empty", "water"],
        "#": ["solid", "ground", "wall"],
        "|": ["passable", "climbable"],
        "B": ["solid","breakable", "penetrable"],
        "t": ["passable", "empty"],
        "A": ["solid", "passable", "penetrable"],
        "M": ["solid", "moving", "penetrable"],
        "D": ["passable", "movable"],
        "W": ["passable", "collectable", "powerup"],
        "w": ["passable", "collectable", "powerup"],
        "L": ["passable", "collectable", "powerup"],
        "l": ["passable", "collectable", "powerup"],
        "+": ["passable", "collectable", "powerup"],
        "*": ["passable", "collectable", "powerup"],
        "U": ["passable", "collectable", "powerup"],
        "Z": ["passable", "collectable", "powerup"],
        "H": ["solid", "hazard"],
        "C": ["passable", "hazard"],
        "p": ["enemy", "damaging", "solid", "moving", "penetrable"],
        "r": ["enemy", "damaging", "ranged"],
        "q": ["enemy", "damaging", "jumping"],
        "o": ["enemy", "damaging", "spawner"],
        "k": ["enemy", "damaging", "spawner"],
        "j": ["enemy", "damaging", "flying"],
        "g": ["enemy", "damaging"],
        "c": ["enemy", "damaging", "ranged"],
        "e": ["enemy", "damaging", "ranged"],
        "m": ["enemy", "damaging", "jumping"],
        "i": ["enemy", "damaging", "ranged"],
        "^": ["enemy", "damaging"],
        "<": ["enemy", "damaging"],
        "f": ["enemy", "damaging", "jumping"],
        "b": ["enemy", "damaging", "flying"],
        "a": ["enemy", "damaging", "ranged"],
        "d": ["enemy", "damaging", "ranged"],
        "h": ["enemy", "damaging", "ranged"]
    }
}


SYSTEM_PROMPT =  """
                    You are a Mega Man captioning agent; given an ascii grid representation of a Mega Man level
                    and a ascii tile set key to go along with it, you must generate EXACTLY FIVE diverse captions 
                    that all describe the level accurately. 

                    RULES:
                    - Your captions should each DISTINCTLY vary in tone, length, wordiness, playfullness, specificity, etc. 
                    while remaining accurate. Make the diversity noticeable, including short and longer captions, playfully
                    descriptive captions and monotone, serious captions, and so on. Keep the longest captions within 3-4 sentences,
                    never overly long. Your shortest captions should be succinct statements about level features/structure.
                    - Do not mention specific tile types in your answer that you see in the tile set (B, p, etc.); 
                    just describe the level with words.
                    - Your captions should primarily focus on level structure, and features in the level, typically 
                    with relative locations, although not explicity required. Mention specific structures/features like
                    platforms, enemies, corridors, etc.
                    - Caption the level like you're writing a prompt to generate it; this means specificity and directness is essential.
                    
                    FORMATTING:
                    - Your response must contain nothing but the five diverse captions.
                    - Put each caption on its own line, with no blank lines between them.
                    - Do not number the captions, add bullets, or write any other text.
                    - You must write exactly FIVE captions; no more, no less.
                    - Do not include any dashes or semicolons. The only punctuation you should 
                    use are commas and periods (, and .)
                 """

def load_dataset(path: str) -> tuple[list[list[str]], str]:
    """
    Load a set of ASCII level scenes for an LLM to caption

    {path} could be:
        - a directory of VGLC-ASCII level files (``*.txt``), one scene per file
        (this is how whole Mega Man levels are loaded today)

        -  a JSON file holding a list of scenes, where each scene is a list of row
        strings, or a dict with a "scene" key holding that list of row strings.

    Either way the return value is a list of scenes, each scene being a list of
    ASCII row strings. Because nothing here assumes a particular scene size, a
    file of smaller ASCII scenes can be dropped in later without code changes.
    """
    # A directory of level files: load_levels reads every *.txt, strips blank
    # lines/trailing whitespace, and returns one list-of-rows per file. Pair each
    # scene with its source filename so callers can label output by level;
    # load_levels and this glob share the same sorted("*.txt") ordering.
    if os.path.isdir(path):
        levels = load_levels(path)
        files = sorted(Path(path).glob("*.txt"))
        return [(level, file.name) for level, file in zip(levels, files)]

    # Otherwise treat it as a JSON file containing a list of pre-built scenes.
    with open(path, "r") as f:
        data = json.load(f)

    scenes = []
    for entry in data:
        # Accept both bare scenes ([rows]) and dataset entries ({"scene": [rows], ...}).
        scenes.append((entry["scene"] if isinstance(entry, dict) else entry, Path(path).name))
    return scenes




def claude_caption(scene: str, game: str = "Mega Man", tileset: dict = MM_TILESET_DICT, model: str = "qwen3.5:9b") -> list[str]:
    """
    Prompt claude (via API) w/ ascii level scene and tileset, return the caption(s) it generates
    """
    tileset_str = json.dumps(tileset, indent=2)

    context = [
        {"role": "user", "content": f"Here is the tileset for {game}:\n{tileset_str}"},
        {"role": "user", "content": f"Level Scene:\n{scene}"},
    ]

    # sup claude
    message = client.messages.create(
                max_tokens=1024,
                system=SYSTEM_PROMPT, # claude requires system prompt to be separated from context block
                messages=context,
                model="claude-sonnet-4-6"
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
            "Given a tileset key and an ASCII level grid, "
            "generate a descriptive yet succinct caption for the level."},
        {"role": "user", "content": f"Here is the tileset for {game}:\n{tileset_str}"},
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
                           help="Directory of VGLC-ASCII level .txt files, or a JSON file of ASCII scenes")
    argparser.add_argument("--game", default="Mega Man",
                           help="Game name passed to the LLM prompt for context")
    argparser.add_argument("--model", default="qwen3.5:9b",
                           help="Local ollama model to prompt for captions")
    argparser.add_argument("--output", default=None,
                           help="Optional path to write the captioned [{scene, caption}] list as JSON")

    return argparser.parse_args()


def main() -> list[list[str]]:

    args = parse_args()

    # load ascii scenes
    scenes = load_dataset(args.levels)

    
    caption_lists = [] # list[list] of caption set for each level
    captioned_dataset = []
    # caption each scene, append back to running lists 
    for i, scene in enumerate(scenes):
        
        scene_str = "\n".join(scene[0]) # first element is the scene
        # currently wired to the claude API version, can also be set to local
        caption_set = claude_caption(scene_str, game=args.game, model=args.model)
        

        print(f"------------------[{scene[1]}] ({i + 1}/{len(scenes)})------------------\n")
        for j, caption in enumerate(caption_set):
            print(f"[Caption {j + 1}/{len(caption_set)}] {caption}\n")

        caption_lists.append(caption_set)
        captioned_dataset.append({"scene": scene, "caption": caption_set[0], "caption1": caption_set[1], "caption2": caption_set[2], "caption3": caption_set[3], "caption4": caption_set[4]})
        
    # save to specified output dir if specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(captioned_dataset, f, indent=2)
        print(f"Captioned dataset saved to {args.output}")

    return caption_lists


if __name__ == "__main__":
    main()
