"""Draw an A* solution path onto a rendered level image.

This mirrors MM-NEAT's MegaManState.vizualizePath / LodeRunnerState.vizualizePath:
render the level, faintly mark every state the search expanded, then trace the
solution path on top. Here we reuse the repo's own tile renderer
(level_dataset.visualize_samples) for the base image and overlay the path with PIL.

The search (astar/AStarSearch.py) hands back a list of *actions*, not states, so we
replay them from the start state via state.get_successor(action) to recover the
(x, y) the agent occupies at each step -- exactly what the Java version does with
current = current.getSuccessor(a).
"""
import os
import sys

# astar/ holds this file and the state files; the repo root holds level_dataset/util.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from PIL import Image, ImageDraw

from level_dataset import visualize_samples
import util.common_settings as common_settings


# game_render name -> (tile count for one-hot, pixel size of each tile)
_RENDER_INFO = {
    "Mario": (common_settings.MARIO_TILE_COUNT, common_settings.MARIO_TILE_PIXEL_DIM),
    "LR": (common_settings.LR_TILE_COUNT, common_settings.LR_TILE_PIXEL_DIM),
    "MM-Simple": (common_settings.MM_SIMPLE_TILE_COUNT, common_settings.MM_TILE_PIXEL_DIM),
    "MM-Full": (common_settings.MM_FULL_TILE_COUNT, common_settings.MM_TILE_PIXEL_DIM),
}

# Overlay colors (RGBA). Visited cells are faint so the path stays readable.
_VISITED_COLOR = (255, 255, 255, 110)
_PATH_COLOR = (220, 30, 30, 255)
_START_COLOR = (40, 200, 60, 255)
_GOAL_COLOR = (40, 120, 255, 255)
_UNREACHABLE_COLOR = (255, 140, 0, 255)   # gold the spawn can't reach, crossed out


def state_xy(state):
    """Return a state's (x, y), tolerating both naming conventions in the state files.

    MarioState/MegaManState expose .x/.y; LodeRunnerState exposes .currentX/.currentY.
    """
    if hasattr(state, "currentX"):
        return state.currentX, state.currentY
    return state.x, state.y


def replay_path(start, solution):
    """Replay the action list from start, collecting the (x, y) at every step.

    Returns a list of (x, y) in the state files' own coordinate space (which, for
    Mario, is the buffered grid -- callers apply the offset when drawing).
    """
    positions = [state_xy(start)]
    current = start
    for action in solution or []:
        current = current.get_successor(action)
        if current is None:        # shouldn't happen for a real solution, but be safe
            break
        positions.append(state_xy(current))
    return positions


def render_scene_image(scene, game_render):
    """Render a raw scene (2D tile-id array) to a PIL image with the game's tiles."""
    if game_render not in _RENDER_INFO:
        raise ValueError(f"Unknown render target {game_render!r}; "
                         f"expected one of {sorted(_RENDER_INFO)}")
    default_classes, _ = _RENDER_INFO[game_render]
    # one_hot needs at least max-id+1 channels; convert_to_level_format argmaxes them back.
    max_id = max((max(row) for row in scene), default=0)
    num_classes = max(default_classes, max_id + 1)

    one_hot = (
        torch.nn.functional.one_hot(torch.tensor(scene, dtype=torch.long), num_classes=num_classes)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    # visualize_samples loads its sprite sheets (mapsheet.png, ...) by repo-root-relative
    # paths, so render from the repo root regardless of where the script was launched.
    cwd = os.getcwd()
    try:
        os.chdir(_REPO_ROOT)
        img = visualize_samples(one_hot, game=game_render)
    finally:
        os.chdir(cwd)
    return img[0] if isinstance(img, list) else img


def visualize_path(scene, game_render, start, solution, visited=None,
                   x_offset=0, y_offset=0, show_visited=True, goal=None):
    """Return a PIL image of the scene with the A* path (and explored cells) drawn on.

    scene        : raw 2D tile-id array (original encoding, used only for rendering)
    game_render  : key into _RENDER_INFO ("Mario", "LR", "MM-Simple", "MM-Full")
    start        : start state the solution is replayed from
    solution     : list of actions returned by the search (None -> draw visited only)
    visited       : iterable of expanded states (drawn faintly); None to skip
    x_offset/y_offset : added to every cell to map state coords -> scene coords
                        (Mario buffers the grid, so it passes -BUFFER_WIDTH for x)
    goal         : explicit (x, y) goal cell to mark in blue (e.g. the placed MM orb), so
                   it shows even when unreachable; None falls back to the end of the path
    """
    _, tile_size = _RENDER_INFO[game_render]
    base = render_scene_image(scene, game_render).convert("RGBA")
    height = len(scene)
    width = len(scene[0])

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def cell_box(x, y):
        return x * tile_size, y * tile_size, (x + 1) * tile_size, (y + 1) * tile_size

    def cell_center(x, y):
        return (x + 0.5) * tile_size, (y + 0.5) * tile_size

    # Faint white X over every expanded state (the search frontier).
    if show_visited and visited:
        line_w = max(1, tile_size // 8)
        for s in visited:
            sx, sy = state_xy(s)
            x, y = sx + x_offset, sy + y_offset
            if not in_bounds(x, y):
                continue
            x0, y0, x1, y1 = cell_box(x, y)
            draw.line([(x0, y0), (x1, y1)], fill=_VISITED_COLOR, width=line_w)
            draw.line([(x1, y0), (x0, y1)], fill=_VISITED_COLOR, width=line_w)

    # Solution path as a polyline through cell centers, clipped to the visible scene.
    positions = replay_path(start, solution)
    drawn = [(x + x_offset, y + y_offset) for (x, y) in positions]
    drawn = [(x, y) for (x, y) in drawn if in_bounds(x, y)]
    if len(drawn) >= 2:
        path_w = max(2, tile_size // 5)
        centers = [cell_center(x, y) for (x, y) in drawn]
        draw.line(centers, fill=_PATH_COLOR, width=path_w, joint="curve")

    # Start (green) and goal (blue) markers.
    def dot(x, y, color):
        cx, cy = cell_center(x, y)
        r = max(2, tile_size // 3)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    if drawn:
        dot(*drawn[0], _START_COLOR)
    # Goal (blue): mark the explicit goal cell if given (e.g. the placed MM orb) so it
    # shows even when the agent can't reach it; otherwise mark the end of the drawn path.
    if goal is not None:
        gx, gy = goal[0] + x_offset, goal[1] + y_offset
        if in_bounds(gx, gy):
            dot(gx, gy, _GOAL_COLOR)
    elif drawn:
        dot(*drawn[-1], _GOAL_COLOR)

    return Image.alpha_composite(base, overlay).convert("RGB")


def visualize_spanning_tree(scene, game_render, start, edges, golds, reachable_golds,
                            visited=None, x_offset=0, y_offset=0, show_visited=True):
    """Return a PIL image of the scene with a reachability spanning tree drawn on

    Draws the BFS spanning tree from the spawn to every reachable gold

    Reachable gold is dotted blue; unreachable gold is crossed out in orange so failures stand out
 
    start: (x, y) spawn cell
    edges: iterable of ((x0, y0), (x1, y1)) parent->child tree branches
    golds: iterable of (x, y) gold cells
    reachable_golds: the subset of golds reachable from the spawn
    visited: reachable (x, y) cells drawn faintly (the reachable region); None to skip
    x_offset/y_offset: added to every cell to map state coords -> scene coords
    """
    _, tile_size = _RENDER_INFO[game_render]
    base = render_scene_image(scene, game_render).convert("RGBA")
    height = len(scene)
    width = len(scene[0])

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def in_bounds(x, y):
        return 0 <= x < width and 0 <= y < height

    def cell_box(x, y):
        return x * tile_size, y * tile_size, (x + 1) * tile_size, (y + 1) * tile_size

    def cell_center(x, y):
        return (x + 0.5) * tile_size, (y + 0.5) * tile_size

    # Faint white X over every reachable cell 
    if show_visited and visited:
        line_w = max(1, tile_size // 8)
        for (vx, vy) in visited:
            x, y = vx + x_offset, vy + y_offset
            if not in_bounds(x, y):
                continue
            x0, y0, x1, y1 = cell_box(x, y)
            draw.line([(x0, y0), (x1, y1)], fill=_VISITED_COLOR, width=line_w)
            draw.line([(x1, y0), (x0, y1)], fill=_VISITED_COLOR, width=line_w)

    # Spanning-tree branches as line segments between adjacent cell centers.
    tree_w = max(2, tile_size // 5)
    for (a, b) in edges:
        ax, ay = a[0] + x_offset, a[1] + y_offset
        bx, by = b[0] + x_offset, b[1] + y_offset
        if in_bounds(ax, ay) and in_bounds(bx, by):
            draw.line([cell_center(ax, ay), cell_center(bx, by)],
                      fill=_PATH_COLOR, width=tree_w)

    def dot(x, y, color):
        cx, cy = cell_center(x, y)
        r = max(2, tile_size // 3)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    def cross(x, y, color):
        x0, y0, x1, y1 = cell_box(x, y)
        w = max(2, tile_size // 5)
        draw.line([(x0, y0), (x1, y1)], fill=color, width=w)
        draw.line([(x1, y0), (x0, y1)], fill=color, width=w)

    # Gold markers: reachable gold dotted blue, unreachable gold crossed orange.
    reachable_set = set(reachable_golds)
    for g in golds:
        gx, gy = g[0] + x_offset, g[1] + y_offset
        if not in_bounds(gx, gy):
            continue
        if g in reachable_set:
            dot(gx, gy, _GOAL_COLOR)
        else:
            cross(gx, gy, _UNREACHABLE_COLOR)

    # Spawn marker (green), drawn last so it stays on top.
    sx, sy = start[0] + x_offset, start[1] + y_offset
    if in_bounds(sx, sy):
        dot(sx, sy, _START_COLOR)

    return Image.alpha_composite(base, overlay).convert("RGB")


def render_info(scene, game_render, info, show_visited=True):
    """Draw whichever overlay that {info} describes onto the scene and return the image.

    {info} is the path_info dict produced by astar_traversability_check.evaluate():
    a "tree" kind (LodeRunner reachability spanning tree) or a "path" kind (a replayed
    A* solution path, the default for Mario/Mega Man)."""
    if info.get("kind") == "tree":
        return visualize_spanning_tree(
            scene, game_render, info["start"], info["edges"],
            info["golds"], info["reachable_golds"], visited=info.get("visited"),
            x_offset=info.get("x_offset", 0), y_offset=info.get("y_offset", 0),
            show_visited=show_visited)
    return visualize_path(
        scene, game_render, info["start"], info["solution"], visited=info.get("visited"),
        x_offset=info.get("x_offset", 0), y_offset=info.get("y_offset", 0),
        show_visited=show_visited, goal=info.get("goal"))
