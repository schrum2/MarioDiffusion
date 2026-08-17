#!/usr/bin/env python
"""Evaluate the A* solvability of every scene in a JSON levels file.

Takes a JSON file of the kind produced by --save_as_json in run_diffusion.py
(or any of the *_create_ascii_captions.save_level_data writers / the MM2
captioned-scene writer) and runs astar_traversability_check.evaluate() (the
no-image traversability check, not astar_path_image) on every scene, then
reports how many/what percentage are solvable.

The only required inputs are the JSON file and the game; the tileset and
tile count are derived from --game the same way run_diffusion.py does it.
"""
import argparse
import json
import sys

import util.common_settings as common_settings
from create_level_json_data import load_tileset
from captions.util import extract_tileset
from astar.astar_traversability_check import evaluate, load_levels, RENDER_GAME_TO_TRAV

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
        "--budget",
        type=int,
        default=100000,
        help="Max states expanded before giving up on a single scene (passed to the A* search)",
    )
    parser.add_argument(
        "--allow_weird_lr",
        action="store_true",
        help="LodeRunner only: allow moving sideways through diggable ground",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-scene printouts; only print the final summary",
    )
    return parser.parse_args()


def evaluate_solvability(json_file, game, output_json=None, quiet=False,
                         budget=100000, allow_weird_lr=False):
    trav_game = RENDER_GAME_TO_TRAV.get(game)
    if trav_game is None:
        raise ValueError(f"Unknown game {game!r}; expected one of {sorted(RENDER_GAME_TO_TRAV)}")

    tileset = GAME_TILESETS[game]
    num_tiles = GAME_TILE_COUNTS[game]

    tile_to_id = load_tileset(tileset)
    _, id_to_char, _, tile_descriptors = extract_tileset(tileset)

    if not quiet:
        print(f"Game: {game}")
        print(f"Tileset: {len(tile_to_id)} tile types from {tileset} (expected {num_tiles} for {game})")

    # load_levels handles a dataset-entry list, a list of raw scenes, or a single
    # scene/entry, mirroring astar_traversability_check.py's own CLI.
    levels = load_levels(json_file)
    if not quiet:
        print(f"Loaded {len(levels)} scene(s) from {json_file}")

    results = []
    solved_count = 0
    failed_count = 0
    for i, (scene, _caption) in enumerate(levels):
        try:
            solved, stats, _info = evaluate(trav_game, scene, id_to_char, tile_descriptors,
                                            budget, allow_weird_lr, visualize=False)
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

    total = len(levels)
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
        evaluate_solvability(args.json_file, args.game, args.output_json, args.quiet,
                             args.budget, args.allow_weird_lr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)