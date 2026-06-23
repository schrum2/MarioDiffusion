"""Calculate the traversability of a level using the ported A* state files.

Given a level JSON file (a dataset entry list, a list of raw scenes, or a single
scene/entry) and a game, this translates the repo's tile encoding into the encoding
each A* state file expects, then runs a search to decide whether the level is
traversable.

Here we map 2D arrays of integer tile IDs that index into a tileset's list of characters.
Each character carries a list of descriptors (e.g. "solid", "passable", "ladder"). We map
descriptors -> the integer encoding used by the corresponding MM-NEAT-derived state file.

Pass --visualize to also draw each solved level's A* path (and explored cells) to a PNG,
mirroring MM-NEAT's vizualizePath; the drawing itself lives in astar_path_visualization.py.
"""
import argparse
import json
import os
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
from MarioState import MarioState, BUFFER_WIDTH
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
    deal with them). 'penetrable' solids (e.g. appearing blocks) are also passable"""
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
def _path_info(start, solution, search, x_offset=0, y_offset=0, goal=None):
    """Bundle the bits the visualizer needs (replay start, path, explored cells).

    goal: explicit (x, y) goal cell to mark even when unreachable (e.g. the placed MM
    orb); None lets the visualizer mark the end of the drawn path instead."""
    return {
        "kind": "path",
        "start": start,
        "solution": solution,
        "visited": search.get_visited(),
        "x_offset": x_offset,
        "y_offset": y_offset,
        "goal": goal,
    }


def mario_traversable(scene, id_to_char, descs, budget, visualize=False):
    grid = translate_scene(scene, id_to_char, descs, mario_tile)
    grid = MarioState.preProcessLevel(grid)          # pipe/bullet fixes (also pads a buffer)
    # Crop the padding back off so the agent is confined to the actual scene: it must not
    width = len(scene[0])
    grid = [row[BUFFER_WIDTH:BUFFER_WIDTH + width] for row in grid]
    height = len(grid)

    # Multi-start: Mario enters from the left edge (scanned from ground -> sky). A cell is only a valid start if he can stand on it
    # If the left edge has no standable cell at all, fall back to the default bottom-left spawn which probably means mario will fall in the void :^( 
    scanner = MarioState(grid, 0, 0, 0)
    def standable(y):
        return (scanner.passable(0, y)
                and scanner.inBounds(0, y + 1)
                and not scanner.passable(0, y + 1))
    starts = [MarioState(grid, 0, 0, y) for y in reversed(range(height)) if standable(y)]
    if not starts:
        starts = [MarioState.from_level(grid)]   # default Mario spawn: bottom-left

    search = AStarSearch(MarioState.moveRight)
    reached = over_budget = False
    solution = None
    winning_start = starts[0]
    for i, start in enumerate(starts):
        try:
            sol = search.search(start, reset=(i == 0), budget=budget)
        except RuntimeError:
            over_budget = True
            break
        if sol is not None:
            reached, solution, winning_start = True, sol, start
            break

    stats = {"reached_goal": reached,
             "path_length": None if solution is None else len(solution),
             "expanded": len(search.get_visited() or [])}
    if over_budget:
        stats["over_budget"] = True
    info = _path_info(winning_start, solution if reached else None, search) if visualize else None
    return reached, stats, info


def lr_traversable(scene, id_to_char, descs, budget, allow_weird=False, visualize=False):
    """Traversable iff every gold is reachable from the spawn.

    Collecting gold never changes the level, so movement legality depends only on the
    player's position (never on which gold has been picked up). That makes each gold's
    reachability independent of collection order; a single BFS from the spawn
    over position space settles every piece of gold at once"""
    grid = translate_scene(scene, id_to_char, descs, lr_tile)
    start = LodeRunnerState.from_level(grid, allowWeirdMoves=allow_weird)
    golds = start.goldLeft
    if not golds: # no gold present -> nothing to do
        return True, {"reached_goal": True, "gold_total": 0, "gold_reachable": 0,
                      "expanded": 0, "note": "no gold in scene"}, None

    parent = start.reachable_tree()                   # BFS spanning tree over reachable cells
    reachable_golds = frozenset(g for g in golds if g in parent)
    reached = reachable_golds == golds

    # Union of the BFS tree branches from the spawn to each reachable gold
    edges = set()
    for g in reachable_golds:
        node = g
        while parent[node] is not None:
            edges.add((parent[node], node))
            node = parent[node]

    stats = {"reached_goal": reached,
             "gold_total": len(golds),
             "gold_reachable": len(reachable_golds),
             "expanded": len(parent)}
    info = None
    if visualize:
        info = {"kind": "tree",
                "start": (start.currentX, start.currentY),
                "edges": edges,
                "golds": list(golds),
                "reachable_golds": reachable_golds,
                "visited": list(parent.keys()),     # the full reachable region
                "x_offset": 0, "y_offset": 0}
    return reached, stats, info



def _in_bounds(grid, cell):
    x, y = cell
    return 0 <= y < len(grid) and 0 <= x < len(grid[0])


def mm_traversable(scene, id_to_char, descs, budget, visualize=False, spawn=None, orb=None):
    """Traversability via the MM-NEAT orb model: drop a spawn and an orb into the scene,
    then run a single A* from the spawn to the orb.

    spawn/orb are optional (x, y) cells - e.g. the user's placed Player Start / Exit Orb
    markers. When given they are used verbatim; otherwise a spawn is auto-placed on the
    low-left and an orb on the right, as before."""
    grid = translate_scene(scene, id_to_char, descs, mm_tile)

    # Place spawn + orb in the grid; from_level then reads them back out as the start
    # position and the heuristic's goal. Explicit (user-placed) cells win; otherwise we
    # auto-place on a natural ledge, carving a pedestal if none exists.
    scanner = MegaManState(grid, 0, 0, (-1, -1), 0, 0)

    if spawn is not None and _in_bounds(grid, spawn):
        grid[spawn[1]][spawn[0]] = mm.MEGA_MAN_TILE_SPAWN
    elif not scanner.placeSpawn():
        scanner.forceSpawn()

    if orb is not None and _in_bounds(grid, orb):
        grid[orb[1]][orb[0]] = mm.MEGA_MAN_TILE_ORB
    elif not scanner.addOrb():
        scanner.forceOrb()

    start = MegaManState.from_level(grid)        # picks up the placed spawn(8) + orb(7)
    if start.x < 0 or start.orb == (-1, -1):
        return False, {"reached_goal": False, "expanded": 0,
                       "note": "could not place spawn/orb"}, None

    search = AStarSearch(MegaManState.orb_heuristic)   # single source, goal = reach the orb
    try:
        solution = search.search(start, budget=budget)
    except RuntimeError:
        return False, {"reached_goal": False, "over_budget": True,
                       "expanded": len(search.get_visited() or [])}, \
               (_path_info(start, None, search, goal=start.orb) if visualize else None)
    reached = solution is not None
    stats = {"reached_goal": reached,
             "path_length": None if solution is None else len(solution),
             "expanded": len(search.get_visited() or [])}
    info = _path_info(start, solution, search, goal=start.orb) if visualize else None
    return reached, stats, info


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


def evaluate(game, scene, id_to_char, descs, budget, allow_weird, visualize=False,
             spawn=None, orb=None):
    """Return (traversable, stats, path_info). path_info is None unless visualize=True
    (or the game short-circuits, e.g. an LR scene with no gold).

    spawn/orb are MM-only optional (x, y) cells used as the start and goal; they are
    ignored for Mario and LR."""
    if game == "Mario":
        return mario_traversable(scene, id_to_char, descs, budget, visualize=visualize)
    if game == "LR":
        return lr_traversable(scene, id_to_char, descs, budget,
                              allow_weird=allow_weird, visualize=visualize)
    if game == "MM":
        return mm_traversable(scene, id_to_char, descs, budget, visualize=visualize,
                              spawn=spawn, orb=orb)
    raise ValueError(f"Unknown game: {game}")


def untraversable_indices(scenes, game, id_to_char, tile_descriptors,
                          budget=100000, allow_weird=False):
    """Return the sorted indices of the scenes that are NOT traversable.

    Intended for filtering un-winnable level slices out of a generated dataset: feed it
    the encoded scenes plus the same tileset mappings they were encoded with, and remove
    the returned indices (in descending order) from the dataset. game is the
    evaluate()-style name ("Mario", "LR", "MM")."""
    bad = []
    for idx, scene in enumerate(scenes):
        ok, _stats, _info = evaluate(game, scene, id_to_char, tile_descriptors,
                                     budget, allow_weird)
        if not ok:
            bad.append(idx)
    return bad


def _render_target(game, tileset_path):
    """Map a game (and tileset) to the name level_dataset.visualize_samples expects."""
    if game == "MM":
        full = os.path.basename(common_settings.MM_FULL_TILESET)
        return "MM-Full" if os.path.basename(tileset_path) == full else "MM-Simple"
    return game  # "Mario" / "LR"


# Render-style game names (as used by run_diffusion and the GUIs) -> the game names
# evaluate() understands. The render name itself doubles as visualize_path's target.
RENDER_GAME_TO_TRAV = {"Mario": "Mario", "LR": "LR",
                       "MM-Simple": "MM", "MM-Full": "MM"}


def astar_path_image(scene, game, id_to_char, tile_descriptors, budget=100000,
                     allow_weird_lr=False, show_visited=True, spawn=None, orb=None):
    """Run A* on a single scene and return (image, traversable, stats).

    game is the render-style name ("Mario", "LR", "MM-Simple", "MM-Full"). image is a
    PIL image of the scene with the path overlaid, or None when there is nothing to
    draw (e.g. an LR scene with no gold).

    spawn/orb are MM-only optional (x, y) cells (the user's placed spawn/exit) used as
    the A* start and goal in place of the auto-placed ones."""
    trav_game = RENDER_GAME_TO_TRAV.get(game)
    if trav_game is None:
        raise ValueError(f"Unknown game {game!r}; expected one of {sorted(RENDER_GAME_TO_TRAV)}")
    ok, stats, info = evaluate(trav_game, scene, id_to_char, tile_descriptors,
                               budget, allow_weird_lr, visualize=True,
                               spawn=spawn, orb=orb)
    if info is None:
        return None, ok, stats
    from astar_path_visualization import render_info
    img = render_info(scene, game, info, show_visited=show_visited)
    return img, ok, stats


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
    parser.add_argument('--visualize', action='store_true',
                        help="Draw the A* solution path over each level and save a PNG")
    parser.add_argument('--image_dir', type=str, default="astar_path_images",
                        help="Directory to write path visualizations into (with --visualize)")
    parser.add_argument('--hide_visited', action='store_true',
                        help="Omit the faint marks for explored (visited) states in the images")
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

    render_info = None
    if args.visualize:
        from astar_path_visualization import render_info
        os.makedirs(args.image_dir, exist_ok=True)
    game_render = _render_target(args.game, tileset_path)

    traversable_count = 0
    untraversable_scenes = [] # stores indexes of untraversable scenes to be returned later, used for filtering weird/untraversable level slices from training sets
    for idx, (scene, _caption) in enumerate(levels):
        ok, stats, path_info = evaluate(args.game, scene, id_to_char, tile_descriptors,
                                        args.budget, args.allow_weird_lr, visualize=args.visualize)
        traversable_count += int(ok)
        
        if not ok:
            untraversable_scenes.append(idx)

        verdict = "TRAVERSABLE" if ok else "NOT traversable"
        detail = ", ".join(f"{k}={v}" for k, v in stats.items())
        print(f"[{idx}] {verdict}  ({detail})")

        if args.visualize and path_info is not None:
            img = render_info(scene, game_render, path_info,
                              show_visited=not args.hide_visited)
            tag = "solved" if ok else "unsolved"
            out_path = os.path.join(args.image_dir, f"level_{idx:04d}_{tag}.png")
            img.save(out_path)
            print(f"      path image -> {out_path}")

    total = len(levels)
    print(f"\n{args.game}: {traversable_count}/{total} traversable "
          f"({100.0 * traversable_count / total:.1f}%)")
    
    return untraversable_scenes



if __name__ == "__main__":
    main()
