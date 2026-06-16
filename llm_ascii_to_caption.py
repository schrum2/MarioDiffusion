import ollama
import json
import argparse

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



def load_dataset(dataset: dict) -> list[dict]:
    """
    Given a JSON dataset as input (formatted as a list of tile-ID level scenes), return a something.
    
    
    """
    pass


def llama_caption(scene: str, game: str) -> str:
    """
    Prompt a local ollama model with something
    """
    tileset = ""
    if game == "MM-Simple" or "MM-Full":
        tileset = MM_TILESET_DICT


    context = [
            { "role": "system", "content": "Given a tileset key and an ASCII level grid,"
            "generate a descriptive yet succint caption for the level" },
            {"role": "user", "content": f"Here is the tileset for {game}: \n + {tileset}"},
            {"role": "user", "content": f"Level Scene: {scene}"}
        ]   
    
    completion = ollama.chat(model="qwen3.5:9b", messages=context)
    caption = completion.message.content
    return caption


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--json", required=True, )


def main():
    
    
    pass