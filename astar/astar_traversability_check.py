"""Calculate the traversability of a level using the ported A* state files.

Given a level JSON file (a dataset entry list, a list of raw scenes, or a single
scene/entry) and a game, this translates the repo's tile encoding into the encoding
each A* state file expects, then runs a search to decide whether the level is
traversable.

Here we map 2D arrays of integer tile IDs that index into a tileset's list of characters.
Each character carries a list of descriptors (e.g. "solid", "passable", "ladder"). We map 
descriptors -> the integer encoding used by the corresponding MM-NEAT-derived state file.
"""
import argparse
import json
import os
import re
import sys

# astar/ holds the state files; the repo root holds captions/ and util/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from captions.util import extract_tileset                            
import util.common_settings as common_settings                       
from AStarSearch import AStarSearch                                  
from MarioState import MarioState                                    
import LodeRunnerState as lr                                         
from LodeRunnerState import LodeRunnerState                          
import MegaManState as mm                                            

MegaManState = mm.MegaManState

# Default tileset per game (the one each dataset is normally created with).
DEFAULT_TILESETS = {
    "Mario": common_settings.MARIO_TILESET,
    "LR": common_settings.LR_TILESET,
    "MM": common_settings.MM_SIMPLE_TILESET,
}


# ---------------------------------------------------------------------------
# Descriptor -> state-file tile encoding, one mapping per game.
# Each takes the set/list of descriptors for a tile and returns the integer the
# corresponding state file expects.
# ---------------------------------------------------------------------------
def mario_tile(descs):
    """MarioState only distinguishes passable vs blocking. Anything walkable-through
    (empty, coins) plus enemies -> passable(2); everything else (ground, pipes, blocks)
    -> solid(0). Those are the only two ids MarioState's passability tables need to
    behave correctly. Enemies are treated as passable (the agent is assumed to deal
    with them) rather than as solid obstacles."""
    return 2 if ("passable" in descs or "enemy" in descs) else 0


def lr_tile(descs):
    """Descriptors -> LodeRunnerState tile ids. Order matters: the specific roles
    (spawn/gold/ladder/rope/enemy/diggable) are checked before the generic
    solid/empty fallback."""
    if "spawn" in descs:
        return lr.LODE_RUNNER_TILE_SPAWN
    if "gold" in descs:
        return lr.LODE_RUNNER_TILE_GOLD
    if "ladder" in descs:
        return lr.LODE_RUNNER_TILE_LADDER
    if "rope" in descs:
        return lr.LODE_RUNNER_TILE_ROPE
    if "enemy" in descs:
        return lr.LODE_RUNNER_TILE_ENEMY
    if "diggable" in descs:
        return lr.LODE_RUNNER_TILE_DIGGABLE
    if "solid" in descs or "ground" in descs:
        return lr.LODE_RUNNER_TILE_GROUND
    return lr.LODE_RUNNER_TILE_EMPTY


def mm_tile(descs):
    """Descriptors -> MegaManState tile ids. Static hazards (spikes, fire pillars)
    stay deadly, but enemies are treated as passable empty (the agent is assumed to
    deal with them). 'penetrable' solids (e.g. appearing blocks) are also passable."""
    if "null" in descs:
        return mm.MEGA_MAN_TILE_NULL          # 9: out-of-bounds padding
    if "climbable" in descs:
        return mm.MEGA_MAN_TILE_LADDER         # 2
    if "water" in descs:
        return mm.MEGA_MAN_TILE_WATER          # 10
    if "breakable" in descs:
        return mm.MEGA_MAN_TILE_BREAKABLE      # 4
    if "enemy" in descs:
        return mm.MEGA_MAN_TILE_EMPTY          # 0: enemies treated as passable
    if "hazard" in descs:
        return mm.MEGA_MAN_TILE_HAZARD         # 3: spikes / fire pillars stay deadly
    if "moving" in descs:
        return mm.MEGA_MAN_TILE_MOVING_PLATFORM  # 5
    if "solid" in descs and "penetrable" not in descs:
        return mm.MEGA_MAN_TILE_GROUND         # 1
    return mm.MEGA_MAN_TILE_EMPTY              # 0 (empty, passable, penetrable solids, items)


def translate_scene(scene, id_to_char, tile_descriptors, tile_fn):
    """Map a scene of repo tile-IDs into the encoding a state file expects."""
    return [
        [tile_fn(tile_descriptors.get(id_to_char[v], set())) for v in row]
        for row in scene
    ]


# ---------------------------------------------------------------------------
# Per-game traversability
# ---------------------------------------------------------------------------
def mario_traversable(scene, id_to_char, descs, budget):
    grid = translate_scene(scene, id_to_char, descs, mario_tile)
    grid = MarioState.preProcessLevel(grid)          # buffer + floor
    start = MarioState.from_level(grid)               # top-left, just above the floor
    search = AStarSearch(MarioState.moveRight)
    try:
        solution = search.search(start, budget=budget)
    except RuntimeError:
        return False, {"reached_goal": False, "over_budget": True,
                       "expanded": len(search.get_visited() or [])}
    return solution is not None, {
        "reached_goal": solution is not None,
        "path_length": None if solution is None else len(solution),
        "expanded": len(search.get_visited() or []),
    }


def lr_traversable(scene, id_to_char, descs, budget, allow_weird=False):
    grid = translate_scene(scene, id_to_char, descs, lr_tile)
    start = LodeRunnerState.from_level(grid, allowWeirdMoves=allow_weird)
    if start.isGoal():                                # no gold present -> nothing to do
        return True, {"reached_goal": True, "path_length": 0, "expanded": 0,
                      "note": "no gold in scene"}
    search = AStarSearch(LodeRunnerState.manhattanToFarthestGold)
    try:
        solution = search.search(start, budget=budget)
    except RuntimeError:
        return False, {"reached_goal": False, "over_budget": True,
                       "expanded": len(search.get_visited() or [])}
    return solution is not None, {
        "reached_goal": solution is not None,
        "path_length": None if solution is None else len(solution),
        "expanded": len(search.get_visited() or []),
    }


_DIR_TO_EDGE = {  # direction word -> (axis, which end)
    "left": ("x", "min"), "right": ("x", "max"),
    "up": ("y", "min"), "down": ("y", "max"),
}


def parse_directions(caption):
    """Pull entrance/exit directions out of an MM caption; default left -> right."""
    entrance, exit_ = "left", "right"
    if caption:
        m = re.search(r"entrance direction (\w+)", caption)
        if m:
            entrance = m.group(1)
        m = re.search(r"exit direction (\w+)", caption)
        if m:
            exit_ = m.group(1)
    return entrance, exit_


def _edge_cells(height, width, direction):
    """Yield (x, y) cells along the edge a direction points to."""
    axis, end = _DIR_TO_EDGE[direction]
    if axis == "x":
        x = 0 if end == "min" else width - 1
        return [(x, y) for y in range(height)]
    y = 0 if end == "min" else height - 1
    return [(x, y) for x in range(width)]


def _edge_target(height, width, direction):
    """The x (or y) coordinate of the edge a direction points to, plus its axis."""
    axis, end = _DIR_TO_EDGE[direction]
    if axis == "x":
        return "x", (0 if end == "min" else width - 1)
    return "y", (0 if end == "min" else height - 1)


def mm_traversable(scene, caption, id_to_char, descs, budget):
    grid = translate_scene(scene, id_to_char, descs, mm_tile)
    height, width = len(grid), len(grid[0])
    entrance, exit_ = parse_directions(caption)

    # A scanner instance just to reuse passable()/inBounds()
    scanner = MegaManState(grid, 0, 0, (-1, -1), 0, 0)
    starts = [
        MegaManState(grid, x, y, (-1, -1), 0, 0)
        for (x, y) in _edge_cells(height, width, entrance)
        if scanner.passable(x, y)
    ]
    if not starts:
        return False, {"reached_goal": False, "expanded": 0,
                       "note": f"no passable cells on {entrance} entrance edge"}

    # Goal: any cell on the exit edge. Heuristic: axis distance to that edge (admissible).
    axis, target = _edge_target(height, width, exit_)
    on_exit_edge = (lambda s: s.x == target) if axis == "x" else (lambda s: s.y == target)
    heuristic = (lambda s: abs(s.x - target)) if axis == "x" else (lambda s: abs(s.y - target))

    # Multi-source: any passable entrance-edge cell could be the true entry point. Run
    # A* from each, sharing one visited set (reset only on the first) so later searches
    # skip already-explored states -- together that's edge-to-edge reachability.
    search = AStarSearch(heuristic)
    reached = over_budget = False
    for i, start in enumerate(starts):
        try:
            solution = search.search(start, reset=(i == 0), budget=budget, is_goal=on_exit_edge)
        except RuntimeError:
            over_budget = True
            break
        if solution is not None:
            reached = True
            break

    stats = {"reached_goal": reached, "expanded": len(search.get_visited() or []),
             "entrance": entrance, "exit": exit_}
    if over_budget:
        stats["over_budget"] = True
    # Known limitation: MegaManState treats open space below as a lethal pit, so an
    # agent seeded on the bottom row (a "down" entrance) cannot move. Vertical
    # corridors also often only make sense with adjacent screens, so down/up results
    # on isolated slices are approximate.
    if "down" in (entrance, exit_) or "up" in (entrance, exit_):
        stats["note"] = "vertical corridor: approximate on isolated slices"
    return reached, stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_levels(path):
    """Return a list of (scene, caption) from a dataset list, list of raw scenes,
    or a single scene/entry."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "scene" in data:
            return [(data["scene"], data.get("caption"))]
        raise ValueError("JSON object has no 'scene' key")

    if isinstance(data, list):
        if not data:
            return []
        first = data[0]
        if isinstance(first, dict):                       # list of dataset entries
            return [(e["scene"], e.get("caption")) for e in data if "scene" in e]
        if isinstance(first, list) and first and isinstance(first[0], list):
            return [(scene, None) for scene in data]      # list of raw scenes
        return [(data, None)]                             # a single raw scene
    raise ValueError("Unsupported JSON structure for a level file")


def evaluate(game, scene, caption, id_to_char, descs, budget, allow_weird):
    if game == "Mario":
        return mario_traversable(scene, id_to_char, descs, budget)
    if game == "LR":
        return lr_traversable(scene, id_to_char, descs, budget, allow_weird=allow_weird)
    if game == "MM":
        return mm_traversable(scene, caption, id_to_char, descs, budget)
    raise ValueError(f"Unknown game: {game}")


def main():
    parser = argparse.ArgumentParser(description="Determine Level Traversability")
    parser.add_argument('--level_json', type=str, required=True,
                        help="Path to the JSON file containing the level(s) to evaluate")
    parser.add_argument('--game', type=str, required=True, choices=["Mario", "LR", "MM"],
                        help="The game the level belongs to; determines how traversability is measured")
    parser.add_argument('--tileset', type=str, default=None,
                        help="Tileset JSON used to encode the scenes (defaults to the game's standard tileset)")
    parser.add_argument('--budget', type=int, default=100000,
                        help="Max states expanded before giving up on a single level")
    parser.add_argument('--allow_weird_lr', action='store_true',
                        help="LodeRunner only: allow moving sideways through diggable ground")
    parser.add_argument('--limit', type=int, default=None,
                        help="Only evaluate the first N levels in the file")
    args = parser.parse_args()

    tileset_path = args.tileset or DEFAULT_TILESETS[args.game]
    if not os.path.isabs(tileset_path) and not os.path.exists(tileset_path):
        tileset_path = os.path.join(_REPO_ROOT, tileset_path)   # resolve repo-relative default
    _, id_to_char, _, tile_descriptors = extract_tileset(tileset_path)

    levels = load_levels(args.level_json)
    if args.limit is not None:
        levels = levels[:args.limit]
    if not levels:
        print("No levels found in file.")
        return

    traversable_count = 0
    for idx, (scene, caption) in enumerate(levels):
        ok, stats = evaluate(args.game, scene, caption, id_to_char, tile_descriptors,
                             args.budget, args.allow_weird_lr)
        traversable_count += int(ok)
        verdict = "TRAVERSABLE" if ok else "NOT traversable"
        detail = ", ".join(f"{k}={v}" for k, v in stats.items())
        print(f"[{idx}] {verdict}  ({detail})")

    total = len(levels)
    print(f"\n{args.game}: {traversable_count}/{total} traversable "
          f"({100.0 * traversable_count / total:.1f}%)")


if __name__ == "__main__":
    main()
