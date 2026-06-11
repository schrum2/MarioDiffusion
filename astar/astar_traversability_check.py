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
def _path_info(start, solution, search, x_offset=0, y_offset=0):
    
    
    
    """Bundle the bits the visualizer needs (replay start, path, explored cells)."""

    return {
        "start": start,
        "solution": solution,
        "visited": search.get_visited(),
        "x_offset": x_offset,
        "y_offset": y_offset,
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
    grid = translate_scene(scene, id_to_char, descs, lr_tile)
    start = LodeRunnerState.from_level(grid, allowWeirdMoves=allow_weird)
    if start.isGoal():                                # no gold present -> nothing to do
        return True, {"reached_goal": True, "path_length": 0, "expanded": 0,
                      "note": "no gold in scene"}, None
    search = AStarSearch(LodeRunnerState.manhattanToFarthestGold)
    info = (lambda sol: _path_info(start, sol, search)) if visualize else (lambda sol: None)
    try:
        solution = search.search(start, budget=budget)
    except RuntimeError:
        return False, {"reached_goal": False, "over_budget": True,
                       "expanded": len(search.get_visited() or [])}, info(None)
    return solution is not None, {
        "reached_goal": solution is not None,
        "path_length": None if solution is None else len(solution),
        "expanded": len(search.get_visited() or []),
    }, info(solution)


_DIR_TO_EDGE = {  # direction word -> (axis, which end)
    "left": ("x", "min"), "right": ("x", "max"),
    "up": ("y", "min"), "down": ("y", "max"),
}


def parse_directions(caption):
    """Pull entrance/exit directions out of an MM caption; default left -> right."""
    entrance, exit = "left", "right"
    if caption:
        m = re.search(r"entrance direction (\w+)", caption)
        if m:
            entrance = m.group(1)
        m = re.search(r"exit direction (\w+)", caption)
        if m:
            exit = m.group(1)
    return entrance, exit


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


def _make_exit_test(grid, direction):
    """Predicate: is a state at the playable edge in "direction"?

    The exit edge is where the screen ends, which is not always the literal grid
    edge: MM scenes are padded to a fixed size with NULL tiles, so the real top of
    an "exit up" level can be an interior row with NULL above it. A state counts as
    exiting when the cell one step further in "direction" is off-grid or NULL;
    i.e. the agent has reached the boundary of the playable area and could leave."""
    height, width = len(grid), len(grid[0])
    null = mm.MEGA_MAN_TILE_NULL

    def beyond_is_offscreen(x, y):
        if direction == "up":
            return y == 0 or grid[y - 1][x] == null
        if direction == "down":
            return y == height - 1 or grid[y + 1][x] == null
        if direction == "left":
            return x == 0 or grid[y][x - 1] == null
        return x == width - 1 or grid[y][x + 1] == null   # right

    return lambda s: beyond_is_offscreen(s.x, s.y)







def mm_traversable(scene, caption, id_to_char, descs, budget, visualize=False):
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
    # Try entrance cells lower on the screen first. The
    # shared visited set means the first start to reach the exit defines the drawn path,
    # so seeding from the floor up makes that path resemble how the screen is actually
    # played. 
    starts.sort(key=lambda s: s.y, reverse=True)
    if not starts:
        return False, {"reached_goal": False, "expanded": 0,
                       "note": f"no passable cells on {entrance} entrance edge"}, None

    # Goal: reach the playable edge in the exit direction (NULL-padding aware; see
    # _make_exit_test). Heuristic: axis distance to the literal edge, which guides the
    # search toward that side (best-first; we only need yes/no reachability, not optimality).
    axis, target = _edge_target(height, width, exit_)
    on_exit_edge = _make_exit_test(grid, exit_)
    heuristic = (lambda s: abs(s.x - target)) if axis == "x" else (lambda s: abs(s.y - target))

    # Multi-source: any passable entrance-edge cell could be an entry point. Run
    # A* from each, sharing one visited set so later searches skip already-explored states
    search = AStarSearch(heuristic)
    reached = over_budget = False
    solution = None
    winning_start = starts[0]            # which start the drawn path replays from
    for i, start in enumerate(starts):
        try:
            solution = search.search(start, reset=(i == 0), budget=budget, is_goal=on_exit_edge)
        except RuntimeError:
            over_budget = True
            break
        if solution is not None:
            reached = True
            winning_start = start
            break

    stats = {"reached_goal": reached, "expanded": len(search.get_visited() or []),
             "entrance": entrance, "exit": exit_}
    if over_budget:
        stats["over_budget"] = True
    # Note: agents on a ladder may now climb off the bottom edge (see MegaManState),
    # but a single screen's vertical corridor often only makes sense together with the
    # screens above/below it, so isolated up/down results stay approximate.
    if "down" in (entrance, exit_) or "up" in (entrance, exit_):
        stats["note"] = "vertical corridor: approximate on isolated slices"
    info = _path_info(winning_start, solution if reached else None, search) if visualize else None
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


def evaluate(game, scene, caption, id_to_char, descs, budget, allow_weird, visualize=False):
    """Return (traversable, stats, path_info). path_info is None unless visualize=True
    (or the game short-circuits, e.g. an LR scene with no gold)."""
    if game == "Mario":
        return mario_traversable(scene, id_to_char, descs, budget, visualize=visualize)
    if game == "LR":
        return lr_traversable(scene, id_to_char, descs, budget,
                              allow_weird=allow_weird, visualize=visualize)
    if game == "MM":
        return mm_traversable(scene, caption, id_to_char, descs, budget, visualize=visualize)
    raise ValueError(f"Unknown game: {game}")


def _render_target(game, tileset_path):
    """Map a game (and tileset) to the name level_dataset.visualize_samples expects."""
    if game == "MM":
        full = os.path.basename(common_settings.MM_FULL_TILESET)
        return "MM-Full" if os.path.basename(tileset_path) == full else "MM-Simple"
    return game  # "Mario" / "LR"


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

    visualize_path = None
    if args.visualize:
        from astar_path_visualization import visualize_path   # lazy: pulls in torch/PIL
        os.makedirs(args.image_dir, exist_ok=True)
    game_render = _render_target(args.game, tileset_path)

    traversable_count = 0
    for idx, (scene, caption) in enumerate(levels):
        ok, stats, path_info = evaluate(args.game, scene, caption, id_to_char, tile_descriptors,
                                        args.budget, args.allow_weird_lr, visualize=args.visualize)
        traversable_count += int(ok)
        verdict = "TRAVERSABLE" if ok else "NOT traversable"
        detail = ", ".join(f"{k}={v}" for k, v in stats.items())
        print(f"[{idx}] {verdict}  ({detail})")

        if args.visualize and path_info is not None:
            img = visualize_path(
                scene, game_render, path_info["start"], path_info["solution"],
                visited=path_info["visited"], x_offset=path_info["x_offset"],
                y_offset=path_info["y_offset"], show_visited=not args.hide_visited,
            )
            tag = "solved" if ok else "unsolved"
            out_path = os.path.join(args.image_dir, f"level_{idx:04d}_{tag}.png")
            img.save(out_path)
            print(f"      path image -> {out_path}")

    total = len(levels)
    print(f"\n{args.game}: {traversable_count}/{total} traversable "
          f"({100.0 * traversable_count / total:.1f}%)")


if __name__ == "__main__":
    main()
