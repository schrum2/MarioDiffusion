#!/usr/bin/env python
"""Evaluate the A* solvability of every scene in a JSON levels file.

Takes a JSON file of the kind produced by --save_as_json in run_diffusion.py
(or any of the *_create_ascii_captions.save_level_data writers / the MM2
captioned-scene writer) and runs the same A* traversability check used
elsewhere in this codebase on every scene, then reports how many/what
percentage are solvable.

The only required inputs are the JSON file and the game; the tileset and
tile count are derived from --game the same way run_diffusion.py does it.
"""
import argparse
import json
import sys

import util.common_settings as common_settings
from create_level_json_data import load_tileset
from captions.util import extract_tileset
from astar.astar_traversability_check import astar_path_image

GAME_TILESETS = {
    "Mario": common_settings.MARIO_TILESET,
    "LR": common_settings.LR_TILESET,
    "MM-Simple": common_settings.MM_SIMPLE_TILESET,
    "MM-Full": common_settings.MM_FULL_TILESET,
    "MMLV": common_settings.MMLV_TILESET,
    "MM2": common_settings.MM2_TILESET,
}

GAME_TILE_COUNTS = {
    "Mario": common_settings.MARIO_TILE_COUNT,
    "LR": common_settings.LR_TILE_COUNT,
    "MM-Simple": common_settings.MM_SIMPLE_TILE_COUNT,
    "MM-Full": common_settings.MM_FULL_TILE_COUNT,
    "MMLV": common_settings.MMLV_TILE_COUNT,
    "MM2": common_settings.MM2_TILE_COUNT,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run A* solvability checks over every scene in a JSON levels file "
        "and report how many/what percentage are solvable."
    )
    parser.add_argument("json_file", type=str, help="Path to the input JSON file (e.g. all_levels.json)")
    parser.add_argument(
        "--game",
        type=str,
        required=True,
        choices=list(GAME_TILESETS.keys()),
        help="Which game the levels belong to (used to derive the tileset and tile count)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to write a detailed per-scene report as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-scene printouts; only print the final summary",
    )
    return parser.parse_args()


def extract_scenes(data):
    """Pull a flat list of scenes (list-of-lists of tile ids) out of the loaded JSON.

    Accepts either a bare list of scenes, or the list-of-dicts format written by
    the save_level_data helpers / the MM2 captioner (each dict holding a "scene" key).
    """
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at the top level, got {type(data).__name__}")

    scenes = []
    for i, entry in enumerate(data):
        if isinstance(entry, dict):
            if "scene" not in entry:
                raise ValueError(f"Entry {i} is a dict but has no 'scene' key: keys={list(entry.keys())}")
            scenes.append(entry["scene"])
        elif isinstance(entry, list):
            scenes.append(entry)
        else:
            raise ValueError(f"Entry {i} is neither a dict nor a list of rows: {type(entry).__name__}")
    return scenes


def evaluate_solvability(json_file, game, output_json=None, quiet=False):
    tileset = GAME_TILESETS[game]
    num_tiles = GAME_TILE_COUNTS[game]

    tile_to_id = load_tileset(tileset)
    id_to_char = {v: k for k, v in tile_to_id.items()}
    _, _, _, tile_descriptors = extract_tileset(tileset)

    if not quiet:
        print(f"Game: {game}")
        print(f"Tileset: {len(tile_to_id)} tile types from {tileset} (expected {num_tiles} for {game})")

    with open(json_file) as f:
        data = json.load(f)
    scenes = extract_scenes(data)
    if not quiet:
        print(f"Loaded {len(scenes)} scene(s) from {json_file}")

    results = []
    solved_count = 0
    failed_count = 0
    for i, scene in enumerate(scenes):
        try:
            _, solved, stats = astar_path_image(scene, game, id_to_char, tile_descriptors)
        except Exception as e:
            failed_count += 1
            if not quiet:
                print(f"Scene {i}: A* check failed ({e})")
            results.append({"index": i, "solved": False, "error": str(e)})
            continue

        if solved:
            solved_count += 1
        result = {"index": i, "solved": bool(solved), "stats": stats}
        results.append(result)
        if not quiet:
            tag = "solved" if solved else "unsolved"
            detail = ", ".join(f"{k}={v}" for k, v in stats.items()) if stats else ""
            print(f"Scene {i}: {tag}" + (f" ({detail})" if detail else ""))

    total = len(scenes)
    pct = (solved_count / total) if total else 0.0
    print(f"\nTotal solvable: {solved_count}/{total}: {pct:.1%}")
    if failed_count:
        print(f"({failed_count} scene(s) raised an error during the A* check and were counted as unsolved)")

    if output_json:
        report = {
            "json_file": json_file,
            "game": game,
            "total": total,
            "solved": solved_count,
            "failed": failed_count,
            "solvable_percentage": pct,
            "results": results,
        }
        with open(output_json, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Wrote detailed report to {output_json}")

    return solved_count, total


if __name__ == "__main__":
    args = parse_args()
    try:
        evaluate_solvability(args.json_file, args.game, args.output_json, args.quiet)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
