"""
This script loads Mega Man levels in VGLC-ASCII format and captions them with an LLM

A loop runs over every scene in the dataset, prompting the LLM with the tileset
key and the ASCII level grid, and assigns each scene a generated caption. The
captions are collected into a list (and optionally written to json) before being
returned.
"""
from pathlib import Path
import os
import json
import argparse

import ollama

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
        "c": "Ranged wall-mounted Blaster enemy",
        "^": "Vertical Adhering Suzy enemy",
        "<": "Horizontal Adhering Suzy enemy",
        "f": "Jumping Big Eye enemy",
        "t": "Secret transparent blocks (looks like regular blocks, but Mega Man can phase through them)",
        "A": "Disappearing/Reappearing blocks (fades in and out)",
        "M": "Moving Platform blocks",
        "D": "Passable Door blocks",
        "W": "Large Weapon Energy power-up",
        "w": "Small Weapon Energy power-up",
        "l": "Small Life Energy power-up",
        "+": "Collectible 1-UP Extra Life Power-up",
        "*": "Collectible Yashichi Power-up",
        "U": "Collectible Magnet Beam Power-up",
        "C": "Hazard Blocks: extends a temporary passable but damaging hazard outward",
        "p": "Ranged Foot Holder enemy, Mega Man can stand and jump from these",
        "r": "Ranged, shielded Sniper Joe enemy",
        "k": "Killer Bomb enemy",
        "g": "Ground Gabyoall enemy",
        "e": "Stationary, ranged Screw Driver enemy",
        "m": "Jumping exploding Bombombomb enemy",
        "i": "Floating, ranged Watcher enemy",
        "b": "Flying Bunby Heli enemy",
        "a": "Stationary, ranged Met enemy",
        "d": "Ranged Pickelman enemy",
        "h": "Crazy Razy enemy",
        "n": "Flying PePe penguin enemy",
        "I": "Changkey vertical fire pillar"
    }
}



OLLAMA_NUM_CTX = 32768
EXPECTED_CAPTIONS = 5
MAX_CAPTION_RETRIES = 20


SYSTEM_PROMPT =  """
You are a Mega Man captioning agent; given an ASCII grid representation of a Mega Man level
and an ASCII tile set key to go along with it, you must generate EXACTLY FIVE diverse captions 
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
- For specificity: one or two captions should contain specific enemy/powerup names, while others can be vague and state enemy class (ranged, ground).
- You must return FIVE distinct captions, each on their own line.
- Don't say things like "This level has..." or "The level features...". Just directly
describe the level itself without mentioning "level".
"""



# Trailing reminder appended as the final (freshest) user msg 
CAPTION_REMINDER = (
    "Reminder: output exactly five captions and nothing else, each on its own line with no "
    "numbering, labels, or blank lines. Break each caption into short '.'-separated phrases, and "
    "make the five intentionally diverse in length, specificity, and tone."
)


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


def filter_tile_set(scene: str, tileset: dict = MM_TILESET_DICT["tiles"]) -> dict:
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



def llm_caption(scene: str, game: str = "Mega Man", tileset: dict = MM_TILESET_DICT, llm: str = "ollama", model: str = "qwen3.5:9b") -> list[str]:


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
            {"role": "user", "content": CAPTION_REMINDER},
        ]

        # sup claude
        message = client.messages.create(
                    max_tokens=2048,
                    system=SYSTEM_PROMPT, # claude requires system prompt to be separated from context block
                    messages=context,
                    model="claude-haiku-4-5"
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
            {"role": "user", "content": CAPTION_REMINDER},
        ]

        # sup mr. altman
        completion = client2.chat.completions.create(
        model="gpt-5.1",
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
    argparser.add_argument("--tileset", default="datasets/MM.json",
                           help="Tileset JSON used to decode integer scenes to ASCII for the prompt "
                                "(must match the tileset the scenes were generated from)")
    argparser.add_argument("--game", default="Mega Man",
                           help="Game name passed to the LLM prompt for context")
    argparser.add_argument("--llm", choices=["claude", "openai", "ollama"], default="ollama",
                           help="The source of the LLM inference used to caption the provided level scenes. The openai and claude choices use APIs, while ollama runs a local model")
    argparser.add_argument("--model", default="qwen3.5:9b",
                           help="Local ollama model to prompt for captions, only used if --llm ollama is argued")
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


    llmstr = args.llm if args.llm != "ollama" else f"{args.llm} - {args.model}"

    # caption each scene, append back to running lists
    for i, (scene, label) in enumerate(scenes):

        # Decode the integer grid to ASCII rows purely to build the prompt; the integer
        # grid itself is what gets stored back in the output.
        scene_str = "\n".join(scene_to_ASCII(scene, id_to_char))

        # Get filtered tileset for current scene
        filtered_tiles = filter_tile_set(scene_str, MM_TILESET_DICT["tiles"])
        
        # assign and collect captions
        caption_set = llm_caption(scene_str, game=args.game, model=args.model, tileset=filtered_tiles, llm=args.llm)
        

        print(f"------------------ [{llmstr}]  [{label}] ({i + 1}/{len(scenes)}) ------------------\n")
        for j, caption in enumerate(caption_set):
            print(f"[Caption {j + 1}/{len(caption_set)}] {caption}\n")

       

        if len(caption_set) != EXPECTED_CAPTIONS:
            print(f"[skip] {label}: got {len(caption_set)} caption(s) instead of {EXPECTED_CAPTIONS}; skipping this scene.\n")
            continue

        caption_lists.append(caption_set)
        # ugly but necessary; want single json object with flat fields scene, cap, cap1, ..., cap4.
        # "scene" holds the original integer tile-id grid
        captioned_dataset.append({"scene": scene, "caption": caption_set[0], "caption1": caption_set[1], "caption2": caption_set[2], "caption3": caption_set[3], "caption4": caption_set[4]})

    # save to specified output dir if specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(captioned_dataset, f, indent=2)
        print(f"Captioned dataset saved to {args.output}")

    return caption_lists


if __name__ == "__main__":
    main()
