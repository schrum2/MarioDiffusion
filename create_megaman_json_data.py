import json
import argparse
from pathlib import Path
import util.common_settings as common_settings
from captions.util import extract_tileset
from create_level_json_data import load_levels
from util.size_utils import level_content_box
from enum import Enum
import os
import sys
import random
import time


#Snap mode: number of null padding rows added on top of each wide (horizontal) scene,
#mirroring the path follower's padding. The screen is nav_height tall and null-free, so a
#wide scene ends up nav_height + SNAP_H_PAD_ROWS tall. Tall scenes use no padding.
SNAP_H_PAD_ROWS = 2


SPAWN_EXIT_CHARS = ('P', 'Z')


#This enum is for the readability of the direction enum
class Axis(Enum):
    VERT=0
    HORIZ=1

#Needed to identify the direction of the sample
class Direction(Enum):
    UP=0, Axis.VERT, -1
    RIGHT=1, Axis.HORIZ, 1
    DOWN=2, Axis.VERT, 1
    LEFT=3, Axis.HORIZ, -1


    #Override new so we can process multiple inputs correctly
    def __new__(cls, value, axis, offset_for_axis):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.axis = axis
        obj.offset_for_axis = offset_for_axis
        return obj #This is the modifier we place on the axis variable to move in that direction

    
    #Move the scene one block in the desired direction
    def move_scene(self, level):
        if self.axis == Axis.VERT: #up/down
            level.move_iter+=1
            level.y_idx += self.offset_for_axis

        if self.axis == Axis.HORIZ: #left/right
            level.move_iter+=1
            level.x_idx += self.offset_for_axis

    
    #Helper method, gets the row or collumn at a given index, depending on axis
    def get_row_or_col(self, level, index):  
        if self.axis == Axis.VERT:
            return level.level[index][level.x_idx:level.x_idx+level.width]
        if self.axis == Axis.HORIZ:
            return [x[index] for x in level.level[level.y_idx:level.y_idx+level.height]] #We need list comprehention to get a vertical slice
    

    #Gets the index of the last row/col of the level sample on the side of the given direction
    def get_index_of_side(self, level): 
        if self.axis == Axis.VERT:
            base = level.y_idx
            modifier=level.height-1
        else:
            base = level.x_idx
            modifier=level.width-1
        
        if self.offset_for_axis==1.0:
            return base+modifier
        return base


    #Check if it's possible to move in a given direction, optionally checking if there's anything blocking MegaMan from moving that way
    def is_possible_to_move_direction(self, level, check_for_walls = False, check_for_possible=True):
        if check_for_possible: #Should always be true exept for when we are recording blocked passages for the json file
            #Check if we're about to move into an out of bounds reigion
            if self.axis==Axis.VERT:
                if level.is_out_of_bounds(y=level.y_idx+self.offset_for_axis):
                    return False
            else:
                if level.is_out_of_bounds(x=level.x_idx+self.offset_for_axis):
                    return False
            #Check to see if moving in the given direction would put us in contact with null chars
            index = self.get_index_of_side(level) + self.offset_for_axis #We want 1 row in that direction
            row = self.get_row_or_col(level, index)

            if any(x in row for x in level.null_chars):
                return False
        
        #Do a second check to see if there is a hole that Mega Man could move through, lower priority than the other two
        if check_for_walls:
            walls_index = self.get_index_of_side(level) #We only want the wall at the end of the row, not the row behind it
            walls_row = self.get_row_or_col(level, walls_index)
            if not any(x not in level.wall_chars for x in walls_row):
                return False
        
        return True #Base case, fires if we're not out of bounds, there's no null ahead, and optionally there's no wall blocking us

def create_tile_to_id(tileset_path, tile_descriptors, new_tileset_dir = 'datasets', group_enemies = True, group_powerups = True, group_empty_tiles = True):
    with open(tileset_path, "r") as f:
        tileset = json.load(f)
        tile_chars = sorted(tileset['tiles'].keys())

        #These variables are used in grouping data later, but defined here to make this optional
        basic_enemy_char = ""
        basic_powerup_char = ""
        basic_empty_tile_char = ""
        enemies = []
        powerups = []
        empty_tiles = []


        #Finding the data to remove
        if group_enemies:
            basic_enemy_char = "a" #Met enemy
            enemies = [x for x in tile_chars if "enemy" in tile_descriptors.get(x)]
        if group_powerups:
            basic_powerup_char = "l" #Small health pack
            powerups = [x for x in tile_chars if "powerup" in tile_descriptors.get(x)]
        if group_empty_tiles:
            basic_empty_tile_char = "-" #Air tile
            empty_tiles = [x for x in tile_chars if ("empty" in tile_descriptors.get(x)) and ("water" not in tile_descriptors.get(x))]

        #Clearing up grouped data, adding basic examples back in
        #Conveyors are left as their own distinct tiles: MM.json has none, and MMLV keeps
        #them as solid conveyor tiles rather than collapsing them into the generic block.
        cleared_list_of_chars = [x for x in tile_chars if x not in enemies+powerups+empty_tiles]
        
        #We do sadly have to do this twice to avoid appending empty chars
        if group_enemies:
            cleared_list_of_chars.append(basic_enemy_char)
        if group_powerups:
            cleared_list_of_chars.append(basic_powerup_char)
        if group_empty_tiles:
            cleared_list_of_chars.append(basic_empty_tile_char)
        
        #Create the basic dictionary
        tile_to_id = {char: idx for idx, char in enumerate(cleared_list_of_chars)}
        id_to_tile = {idx: char for char, idx in tile_to_id.items()}

        #Create a new tileset to match these tiles. We also persist the *actual* id
        #assignment order used to encode scene data, since sort_keys=True below alphabetizes
        #the "tiles" section for human readability and no longer reflects the real ids.
        output = os.path.join(new_tileset_dir, "MM-simple-tileset.json")
        tile_dict = {tile: sorted(list(tile_descriptors.get(tile))) for tile in tile_to_id}
        tile_dict = {"tiles" : tile_dict, "tile_to_id": dict(tile_to_id)}

        with open(output, 'w') as f:
            json.dump(tile_dict, f, indent=4, sort_keys=True)

        #Add in the old tiles to allow for encoding of everything
        tile_to_id_enemies = {char: tile_to_id[basic_enemy_char] for char in enemies}
        tile_to_id_powerups = {char: tile_to_id[basic_powerup_char] for char in powerups}
        tile_to_id_null_tiles = {char: tile_to_id[basic_empty_tile_char] for char in empty_tiles}
        tile_to_id.update(tile_to_id_enemies)
        tile_to_id.update(tile_to_id_powerups)
        tile_to_id.update(tile_to_id_null_tiles)
        return tile_to_id, id_to_tile


def parse_args():
    parser = argparse.ArgumentParser(description="Create level json files for megaman")
    
    parser.add_argument('--tileset', default='datasets/MM.json', help='Path to the tile set JSON')
    parser.add_argument('--levels', default='../TheVGLC/MegaMan/Enhanced', help='Directory containing level text files')
    parser.add_argument('--output', required=True, help='Path to the output directory')
    parser.add_argument('--no_filter', action='store_true', help='Disable all quality/playable-area filtering (including the A* traversability filter); write every extracted scene to --output.')
    parser.add_argument('--target_height', type=int, default=common_settings.MEGAMAN_WIDTH, help='Output scene height (e.g., 16 or 32). Navigation still uses the screen height for path mode.')
    parser.add_argument('--target_width', type=int, default=common_settings.MEGAMAN_WIDTH, help='Output scene width (e.g., 16 or 32). Navigation still uses the screen width for path mode.')
    parser.add_argument('--faithful_vertical', action='store_true', help='Fill the rows above the navigation window with real level content instead of null padding (auto-enabled when --target_height exceeds the default square).')
    parser.add_argument('--group_encodings', action='store_true', help='Group the tile encodings by type to reduce the total number')
    parser.add_argument('--keep_spawn_exit', action='store_true', help="Keep the player spawn ('P') and exit orb ('Z') tiles in the output scenes. By default these markers are stripped (encoded as air) since they are level metadata, not geometry to be learned.")
    #The A* traversability filter is on by default now (also feeds the low-content check in apply_filters); --no_traversable_filter turns the hard filter off.
    parser.add_argument('--no_traversable_filter', dest='traversable_only', action='store_false', default=True, help='Disable filtering out A*-untraversable scenes (this filter is ON by default). The A* path length is still computed for the low-content rescue check regardless.')
    parser.add_argument('--budget', type=int, default=100000, help='A* state-expansion budget per scene used by the traversability check (higher = more thorough, slower)')
    parser.add_argument('--scan_mode', default='path', choices=['path', 'sliding_window', 'snap', 'screen_grid', 'whole'], help='How to extract samples: path follower (default), sliding window, snap (variable-dimension wide+tall scans that snap to valid content), screen_grid (tile the level into a 2x2 (or screens_x x screens_y) grid of Mega Man Maker screens, keeping windows where at least 3/4 of the quadrants are occupied, with null padding on top to reach target_height -- e.g. target 32x32 -> 32x28 content + 4 pad rows), or whole (one sample = the entire level trimmed to its content bounding box, kept at its natural variable size)')
    parser.add_argument('--direction_captions', action='store_true', help='Whether to include entrance/exit directional captions when creating datasets; defaults to False')
    parser.add_argument('--stride_y', type=int, default=1, help='How far the sliding window moves in the vertical direction during level scanning (sliding_window/snap modes only; must be >= 1)')
    parser.add_argument('--stride_x', type=int, default=1, help='How far the sliding window moves in the horizontal direction during level scanning (sliding_window/snap modes only; must be >= 1)')
    parser.add_argument('--max_enemies', type=int, default=8, help='Filter out scenes with more than this many enemy tiles. Omit to disable.')
    parser.add_argument('--include_moving_ground', action='store_true', help='Include scenes containing moving-ground/platform tiles (e.g. "M"). By default these are excluded since their motion is not represented in the static scene graphics.')
    parser.add_argument('--min_content_pct', type=float, default=7, help='Filter out scenes where less than this percent of tiles are real content (not empty/passable/null). E.g. 15 requires at least 15%% non-empty tiles.')
    parser.add_argument('--min_playable_tiles', type=int, default=10, help='Filter out scenes where a flood fill starting from the border reaches fewer than this many open (non-wall, non-null) tiles -- i.e. scenes with almost no playable area connected to their edges. Default 10 (out of 224 in a 16x14 scene); set to 0 to disable.')
    parser.add_argument('--min_content_path_len', type=int, default=14, help='Low-content rescue: a scene flagged as low-content is kept anyway if the A* agent can traverse it along a path at least this many steps long (default 14, a little under the scene width). This spares genuinely sparse-but-playable scenes (e.g. spread-out parkour rooms). Set to 0 to disable the rescue.')
    parser.add_argument('--limit', type=int, default=None, help='Cap the number of kept samples saved to --output. Applied after filtering; the first N kept samples are written. Omit for no cap.')

    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.stride_x < 1 or args.stride_y < 1:
        parser.error("--stride_x and --stride_y must be >= 1")
    return args

def border_flood_fill_count(scene, blocking_ids, void_ids=frozenset()):
    """ 
    Count how many tiles are reachable by a fill seeded from
    every open tile the player could enter the scene from.

    Used to filter out scenes that are completely blocked in
    """
    height = len(scene)
    width = len(scene[0]) if height else 0
    if not width:
        return 0

    visited = [[False] * width for _ in range(height)]
    stack = []

    def seed(x, y):
        if not visited[y][x] and scene[y][x] not in blocking_ids:
            visited[y][x] = True
            stack.append((x, y))

    # Literal grid border: ceiling, floor, and the left/right columns.
    for x in range(width):
        seed(x, 0)
        seed(x, height - 1)
    for y in range(height):
        seed(0, y)
        seed(width - 1, y)

    # Openings onto the off-screen void: any open tile next to null padding is also an
    # entrance (e.g. an open ceiling sitting under padding rows, or padding along a side).
    if void_ids:
        for y in range(height):
            for x in range(width):
                if scene[y][x] in void_ids:
                    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                        if 0 <= nx < width and 0 <= ny < height:
                            seed(nx, ny)

    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx] \
                    and scene[ny][nx] not in blocking_ids:
                visited[ny][nx] = True
                stack.append((nx, ny))
    return count


#Ordered so that when a scene fails more than one check, we report the most fundamental
#reason first (can't be traversed at all -> no playable area -> too dense with enemies ->
#static-motion tiles -> not enough real content).
FILTER_REASONS = ("not_traversable", "insufficient_playable_area",
                  "too_many_enemies", "moving_ground", "low_content")

def apply_filters(all_samples, id_to_char, tile_descriptors, *, traversable_only=True,
                  budget=100000, max_enemies=8, exclude_moving_ground=True,
                  min_content_pct=15, min_playable_tiles=10, min_content_path_len=14):
    """Split all_samples into (kept, filtered). Each filtered sample is returned as a copy
    tagged with a 'filter_reasons' key: the list of *every* reason it was cut (a subset of
    FILTER_REASONS, in priority order), not just the first.

    Filters, in priority order:
      not_traversable          -- A* orb check fails (only cuts when traversable_only is set)
      insufficient_playable_area -- border-originating flood fill reaches < min_playable_tiles
      too_many_enemies         -- more than max_enemies enemy tiles
      moving_ground            -- contains moving-ground/platform tiles (static graphics can't show motion)
      low_content              -- less than min_content_pct real (non-empty/null) content
    Any threshold passed as None (or, for min_playable_tiles/min_content_path_len, <= 0)
    disables that check.

    The A* pass feeds both not_traversable and a low_content rescue: a scene flagged
    low_content but which the A* agent can actually traverse along a path >= min_content_path_len
    steps is a genuinely sparse-but-playable scene (e.g. a spread-out parkour room), so the
    low_content reason is dropped and, absent any other reason, the scene is kept. A* is the
    expensive step, so it only runs where it can change the outcome: on every scene when
    traversable_only is set, but otherwise only on scenes whose sole reason is low_content
    (the only case the rescue can affect)."""
    wall_ids = {tid for tid, ch in id_to_char.items()
                if "solid" in tile_descriptors.get(ch, []) and "penetrable" not in tile_descriptors.get(ch, [])}
    null_ids = {tid for tid, ch in id_to_char.items() if "null" in tile_descriptors.get(ch, [])}
    #Tiles the flood fill treats as impassable: solid walls plus out-of-bounds null padding.
    blocking_ids = wall_ids | null_ids

    moving_ids = {tid for tid, ch in id_to_char.items() if "moving" in tile_descriptors.get(ch, [])} \
        if exclude_moving_ground else set()
    enemy_ids = {tid for tid, ch in id_to_char.items() if "enemy" in tile_descriptors.get(ch, [])} \
        if max_enemies is not None else set()
    #"Empty" content = tiles tagged empty (air/water/etc.) or null (out of bounds padding),
    #i.e. tiles that don't represent anything to interact with. Everything else (solid,
    #hazard, enemy, climbable, powerup) counts as real content.
    empty_ids = {tid for tid, ch in id_to_char.items()
                 if "empty" in tile_descriptors.get(ch, []) or "null" in tile_descriptors.get(ch, [])} \
        if min_content_pct is not None else set()

    #First pass: collect the cheap (non-A*) reasons for every scene, in FILTER_REASONS
    #(priority) order but excluding the A*-derived not_traversable, which is prepended later.
    base_reasons = []
    for s in all_samples:
        scene = s["scene"]
        total = len(scene) * len(scene[0])
        reasons = []
        if min_playable_tiles and min_playable_tiles > 0 \
                and border_flood_fill_count(scene, blocking_ids, null_ids) < min_playable_tiles:
            reasons.append("insufficient_playable_area")
        if max_enemies is not None \
                and sum(1 for row in scene for tile in row if tile in enemy_ids) > max_enemies:
            reasons.append("too_many_enemies")
        if exclude_moving_ground and any(tile in moving_ids for row in scene for tile in row):
            reasons.append("moving_ground")
        if min_content_pct is not None and total:
            empty_count = sum(1 for row in scene for tile in row if tile in empty_ids)
            content_pct = 100.0 * (total - empty_count) / total
            if content_pct < min_content_pct:
                reasons.append("low_content")
        base_reasons.append(reasons)

    #The A* pass is the expensive part, so only run it where its result can change the
    #outcome. When traversable_only is on, every scene needs it (A* decides not_traversable,
    #and its path length can also rescue a low_content scene). When the traversability filter
    #is off, A* can *only* rescue a scene whose sole problem is low_content -- a scene with any
    #other reason stays filtered regardless -- so we run it just on those. evaluate() is the
    #traversability dispatcher; call it directly to get both the reached flag and path length.
    if traversable_only:
        astar_indices = range(len(all_samples))
    else:
        astar_indices = [i for i, r in enumerate(base_reasons) if r == ["low_content"]]

    traversable_flags = {}   # idx -> reached-goal bool (only for scenes A* was run on)
    path_lengths = {}        # idx -> A* solution length (or None)
    if astar_indices:
        astar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astar")
        if astar_dir not in sys.path:
            sys.path.insert(0, astar_dir)
        from astar_traversability_check import evaluate
        for i in astar_indices:
            ok, stats, _info = evaluate("MM", all_samples[i]["scene"], id_to_char,
                                        tile_descriptors, budget, False)
            traversable_flags[i] = ok
            path_lengths[i] = stats.get("path_length")

    kept = []
    filtered = []
    #Per-reason tallies: a scene can trip several filters, so these can sum to more than the
    #number of filtered scenes.
    reason_counts = {r: 0 for r in FILTER_REASONS}

    for idx, s in enumerate(all_samples):
        #Collect *every* reason this scene trips, in FILTER_REASONS (priority) order.
        reasons = list(base_reasons[idx])

        #not_traversable is the highest-priority reason, so it goes at the front.
        if traversable_only and not traversable_flags[idx]:
            reasons.insert(0, "not_traversable")

        #Low-content rescue: if the only thing wrong is sparseness, but the A* agent can
        #actually traverse this scene along a long path (>= min_content_path_len steps, a bit
        #under the scene width), it is a legitimately sparse-but-playable scene rather than
        #empty filler -- drop the low_content reason. Any other reason still cuts it. (A* was
        #only run for these scenes when the traversability filter is off, hence the .get.)
        path_len = path_lengths.get(idx)
        if "low_content" in reasons and traversable_flags.get(idx, False) \
                and min_content_path_len and min_content_path_len > 0 \
                and path_len is not None and path_len >= min_content_path_len:
            reasons.remove("low_content")

        if not reasons:
            kept.append(s)
        else:
            for r in reasons:
                reason_counts[r] += 1
            filtered.append({**s, "filter_reasons": reasons})

    total_samples = len(all_samples)
    print(f"Filtering: kept {len(kept)}/{total_samples}, removed {len(filtered)} "
          f"({', '.join(f'{r}={reason_counts[r]}' for r in FILTER_REASONS)}).")
    return kept, filtered

def extract_mmlv_id(filename):
    """Pull the Mega Man Maker level ID out of an MMLV-derived ASCII filename.
    If the filename stem is entirely digits it is an MMLV level and we return the ID 
    as an int; otherwise it is not MMLV-derived and we return None so no mmlvID is logged."""
    stem = Path(filename).stem
    return int(stem) if stem.isdigit() else None


def main():

    args = parse_args()

    levels = load_levels(args.levels)
    #Mirrors load_levels' own sorted glob so level_files[i] is guaranteed to be the same
    #file that produced levels[i] -- gives us a real filename to tag samples with, without
    #needing to modify load_levels itself.
    level_files = sorted(Path(args.levels).glob("*.txt"))

    #Per-level metadata (name/author/downloads/likes/dislikes) fetched at download time by
    #Bulk_Download.py and keyed by MMLV level id (string). Read from the single master sidecar
    #at the repo-wide constant path, looked up by mmlv_id per level below, and attached to
    #every sample so it survives into the training data. Absent for the VGLC Enhanced set
    #(non-numeric filenames) or before any download, so a missing file is not an error.
    metadata_path = common_settings.MEGAMAN_METADATA_PATH
    level_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            level_metadata = json.load(f)
        print(f"Loaded metadata for {len(level_metadata)} levels from {metadata_path}")
    else:
        print(f"No level metadata found at {metadata_path}; samples will have metadata=None")
    _, id_to_char, tile_to_id, tile_descriptors = extract_tileset(args.tileset)
    null_chars = [key for key, value in tile_descriptors.items() if 'null' in value]
    wall_chars = [key for key, value in tile_descriptors.items() if (('solid' in value) and ('penetrable' not in value))]
    
    if args.group_encodings:
        tile_to_id, id_to_char = create_tile_to_id(args.tileset, tile_descriptors, new_tileset_dir=os.path.dirname(args.output))

    # Strip the spawn/exit markers, and remap their chars to the air id unless --keep_spawn_exit
    # find_start still scans the raw 'P' char, so spawn detection is unaffected
    if not args.keep_spawn_exit:
        air_id = tile_to_id.get('-')
        if air_id is None:
            air_id = next((tid for ch, tid in tile_to_id.items()
                           if 'empty' in tile_descriptors.get(ch, []) and 'water' not in tile_descriptors.get(ch, [])), 0)
        strip_chars = {ch for ch in tile_to_id if 'spawn' in tile_descriptors.get(ch, [])}
        strip_chars.update(ch for ch in SPAWN_EXIT_CHARS if ch in tile_to_id)
        for ch in strip_chars:
            tile_to_id[ch] = air_id
        if strip_chars:
            print(f"Stripping spawn/exit tiles {sorted(strip_chars)} -> air id {air_id} "
                  f"(pass --keep_spawn_exit to retain them)")

    #We literally only need level overrides for 1-7, every other level parses as expected
    overrides_1_7 = [120, 121, 122, 123, 182] #Needed to avoid an early turn leading to a split path, and to prevent the level from turning back around to go back to the start

    nav_width = common_settings.MEGAMAN_WIDTH
    nav_height = common_settings.MEGAMAN_HEIGHT

    faithful_vertical = args.faithful_vertical or (args.target_height > nav_width)

    direction_captions = args.direction_captions

    all_samples = []
    seen_samples = set()
    duplicates_removed = 0

    for i in range(len(levels)):
        #Source filename for this level, tagged onto every sample below so a bad scene
        #can be traced back to exactly where it came from. Falls back to the index if
        #level_files is ever shorter than levels for some reason.
        source_level_name = level_files[i].name if i < len(level_files) else f"level_{i}"
        mmlv_id = extract_mmlv_id(source_level_name)
        #Metadata record for this level (None for non-MMLV levels or ids missing from the
        #sidecar); attached to every sample cut from this level below.
        mmlv_meta = level_metadata.get(str(mmlv_id)) if mmlv_id is not None else None

        try:
            if args.scan_mode == 'snap':
                #Two scans producing a mix of wide and tall scenes that snap to real
                #content (see snap_window_samples). Both scan for fully null-free screens.
                #Wide scenes are a target_width x nav_height null-free screen with
                #SNAP_H_PAD_ROWS rows of null padding added on top (matching the path
                #follower), for a final height of nav_height + SNAP_H_PAD_ROWS.
                h_samples, h_json, h_coords = snap_window_samples(
                    levels[i], tile_to_id, args.target_width, nav_height, null_chars,
                    top_pad=args.target_height % 14, x_stride=args.stride_x, y_stride=args.stride_y,
                    direction_captions=direction_captions
                )
                #Tall scenes are a nav_width x target_height null-free screen with no
                #padding. But when target_height is no taller than the standard padded
                #square (nav_height + SNAP_H_PAD_ROWS == 16) -- including the default case,
                #where the user passes no --target_height -- there is nothing "tall" to
                #capture, so the vertical scan instead emits the same padded 16x16 scene as
                #the wide scan: a nav_width x nav_height null-free screen with SNAP_H_PAD_ROWS
                #null rows on top, keeping it consistent with the wide and path scans.
                if args.target_height <= nav_height + SNAP_H_PAD_ROWS:
                    v_screen_height, v_top_pad = nav_height, SNAP_H_PAD_ROWS
                else:
                    v_screen_height, v_top_pad = args.target_height, 0
                v_samples, v_json, v_coords = snap_window_samples(
                    levels[i], tile_to_id, nav_width, v_screen_height, null_chars,
                    top_pad=v_top_pad, x_stride=args.stride_x, y_stride=args.stride_y,
                    direction_captions=direction_captions
                )
                samples = h_samples + v_samples
                json_caption_data = h_json + v_json
                source_coords = h_coords + v_coords
                scan_mode_tags = (["snap_wide"] * len(h_samples)) + (["snap_tall"] * len(v_samples))
            elif args.scan_mode == 'sliding_window':
                samples, json_caption_data, source_coords = sliding_window_samples(
                    levels[i], tile_to_id, nav_width, nav_height, null_chars,
                    out_width=args.target_width, out_height=args.target_height,
                    x_stride=args.stride_x, y_stride=args.stride_y
                )
                scan_mode_tags = ["sliding_window"] * len(samples)
            elif args.scan_mode == 'whole':
                # One sample = the entire level trimmed to its content bounding box,
                # kept at its natural (variable) size. Air/null border is cut off;
                # ragged rows are right-filled with the null tile so any out-of-region
                # cell is the void tile (matching the analyze_level_dimensions measure).
                null_char = null_chars[0] if null_chars else '@'
                empty_chars = "-" + "".join(null_chars)
                box = level_content_box(levels[i], empty_chars=empty_chars, fill=null_char)
                if not box:
                    raise ValueError("level has no content to extract")
                null_id = tile_to_id.get(null_char, 0)
                encoded = [[tile_to_id.get(ch, null_id) for ch in row] for row in box]
                samples = [encoded]
                json_caption_data = [None]
                source_coords = [(0, 0)]
                scan_mode_tags = ["whole"]
            elif args.scan_mode == 'screen_grid':
                #32x32-style scenes built from the Mega Man Maker screen grid. Each MMLV
                #screen is nav_width x nav_height (16x14), and the converter fills whole
                #screens as either all-null ('@', a screen that doesn't exist) or real
                #content. We tile the level with a window spanning screens_x x screens_y
                #screens -- the non-padded content region -- moving in screen-sized strides
                #(pass --stride_x 16 --stride_y 14 for the standard overlapping scan). A
                #window is kept only if at least 3/4 of its quadrants are occupied (no more
                #than 1/4 of the screen-sized quadrants are null), then top_pad null rows are
                #added on top so the final scene reaches target_height. Default 32x32: 2x2
                #screens = 32x28 content + 4 null pad rows.
                screen_w, screen_h = nav_width, nav_height
                screens_x = max(1, args.target_width // screen_w)
                screens_y = max(1, args.target_height // screen_h)
                content_h = screens_y * screen_h
                top_pad = args.target_height - content_h
                if top_pad < 0:
                    raise ValueError(
                        f"target_height ({args.target_height}) is smaller than "
                        f"{screens_y} screens ({content_h}); cannot pad a negative number of rows"
                    )
                total_quadrants = screens_x * screens_y
                #"No more than 1/4 of the quadrants null" -> for a 2x2 grid this allows 1
                #null quadrant, i.e. requires at least 3 of 4 occupied.
                min_occupied = total_quadrants - (total_quadrants // 4)
                samples, json_caption_data, source_coords = screen_grid_samples(
                    levels[i], tile_to_id, screen_w, screen_h, screens_x, screens_y,
                    top_pad, null_chars, min_occupied,
                    x_stride=args.stride_x, y_stride=args.stride_y
                )
                scan_mode_tags = ["screen_grid"] * len(samples)
            elif i == 7:
                samples, json_caption_data, source_coords = parse_level(
                    tile_to_id, levels[i], nav_width, nav_height,
                    null_chars, wall_chars,
                    out_width=args.target_width,
                    out_height=args.target_height,
                    faithful_vertical=faithful_vertical,
                    print_at_corners=False,
                    change_direction_overrides=overrides_1_7,
                    direction_captions=direction_captions
                )
                scan_mode_tags = ["path"] * len(samples)
            else:
                samples, json_caption_data, source_coords = parse_level(
                    tile_to_id, levels[i], nav_width, nav_height,
                    null_chars, wall_chars,
                    out_width=args.target_width,
                    out_height=args.target_height,
                    faithful_vertical=faithful_vertical,
                    print_at_corners=False,
                    direction_captions=direction_captions
                )
                scan_mode_tags = ["path"] * len(samples)

        except ValueError as e:
            print(f"Skipping level {i}: {e}")
            continue

        print(f"Level {i} parsed successfully: {len(samples)} samples")

        for sample, json_data, (src_x, src_y), mode_tag in zip(samples, json_caption_data, source_coords, scan_mode_tags):

            #json_data is None when directional captions are off (see parse_level /
            #snap_window_samples). Fall back to sentinel directions so the dedup key
            #stays a homogeneous 3-tuple instead of subscripting None.
            sample_key = tuple(tuple(row) for row in sample)
            if json_data is None:
                key = (sample_key, None, None)
            else:
                key = (
                    sample_key,
                    json_data["entrance_direction"],
                    json_data["exit_direction"]
                )

            if key in seen_samples:
                duplicates_removed += 1
                continue

            seen_samples.add(key)

            all_samples.append({
                "scene": sample,
                "data": json_data,
                "source_level": source_level_name,
                "source_x": src_x,
                "source_y": src_y,
                "scan_mode": mode_tag,
                "mmlvID": mmlv_id,
                "metadata": mmlv_meta
            })

    print(f"Removed {duplicates_removed} duplicate samples")
    print(f"Final dataset size: {len(all_samples)}")

    #Time the filtering + save stage. The A* traversability pass runs on every scene here
    #and dominates the runtime, so report how long it takes to produce the final datasets.
    filter_start = time.perf_counter()

    # The quality/traversability filters are tuned for small navigation-sized windows
    # (e.g. max_enemies=8, min_content_pct=15) and would delete entire levels in whole
    # mode, so they are not applied there. A whole level is kept as-is.
    filtered_samples = []
    if args.scan_mode == 'whole':
        print("scan_mode=whole: skipping window-oriented quality/traversability filters")
    elif args.no_filter:
        print("--no_filter: skipping all quality/playable-area filtering")
    else:
        all_samples, filtered_samples = apply_filters(
            all_samples, id_to_char, tile_descriptors,
            traversable_only=args.traversable_only, budget=args.budget,
            max_enemies=args.max_enemies, exclude_moving_ground=not args.include_moving_ground,
            min_content_pct=args.min_content_pct, min_playable_tiles=args.min_playable_tiles,
            min_content_path_len=args.min_content_path_len,
        )

    #Cap the number of saved samples if requested. Applied after filtering so the limit
    #counts only kept (quality-passing) samples; the first N are written.
    if args.limit is not None and len(all_samples) > args.limit:
        print(f"--limit: capping saved samples at {args.limit} (from {len(all_samples)})")
        all_samples = all_samples[:args.limit]

    output = args.output
    with open(output, 'w') as f:
        json.dump(all_samples, f, indent=2)
    print(f"Saved {len(all_samples)} kept samples to {output}")

    # Write the filtered-out scenes (each tagged with its filter_reasons) to a sibling file
    # so they can be inspected/audited instead of being silently discarded. Named by
    # tacking "-filtered" onto --output, e.g. MM_Levels.json -> MM_Levels-filtered.json.
    if filtered_samples:
        stem, ext = os.path.splitext(output)
        filtered_output = f"{stem}-filtered{ext or '.json'}"
        with open(filtered_output, 'w') as f:
            json.dump(filtered_samples, f, indent=2)
        print(f"Saved {len(filtered_samples)} filtered samples to {filtered_output}")

    elapsed = time.perf_counter() - filter_start
    print(f"Filtering + save took {elapsed:.1f}s ({elapsed / 60:.1f} min)")

def sliding_window_samples(level, tile_to_id, width, height, null_chars, out_width=None, out_height=None, x_stride = 1, y_stride = 1):
    out_width = out_width or width
    out_height = out_height or height
    null = null_chars[0]
    level_height = len(level)
    level_width = len(level[0])

    samples = []
    json_caption_data = []
    source_coords = []

    for y in range(0, level_height - height + 1, y_stride):
        for x in range(0, level_width - width + 1, x_stride):
            window = [level[y+r][x:x+width] for r in range(height)]

            at_count = sum(1 for row in window for c in row if c == '@')
            total = len(window) * len(window[0])
            if at_count / total > 0.60:
                continue

            # Reject windows that contain a fully-void column or row running
            # through them. A legitimate in-room sample should never have a
            # complete blank column/row, since real rooms are walkable
            # (filled with '-') everywhere they exist. A full void column/row
            # means the window straddles a real room and a real empty gap
            # between two disconnected rooms.
            has_full_void_column = any(
                all(window[r][c] == '@' for r in range(height))
                for c in range(width)
            )
            has_full_void_row = any(
                all(c == '@' for c in row)
                for row in window
            )
            if has_full_void_column or has_full_void_row:
                continue

            col_offset = (out_width - width) // 2
            row_offset = out_height - height
            level_x0 = x - col_offset
            level_y0 = y - row_offset

            sample = []
            for r in range(out_height):
                ly = level_y0 + r
                out_row = []
                for c in range(out_width):
                    lx = level_x0 + c
                    if 0 <= ly < level_height and 0 <= lx < level_width:
                        out_row.append(level[ly][lx])
                    else:
                        out_row.append(null)
                sample.append(out_row)

            encoded = [[tile_to_id.get(ch, tile_to_id.get(null, 0)) for ch in row] for row in sample]
            samples.append(encoded)
            json_caption_data.append({"entrance_direction": "RIGHT", "exit_direction": "RIGHT"})
            #Source tile coords (top-left of the scanned window, before out_width/out_height
            #padding) so a generated sample can be traced back to its exact spot in the
            #original level for debugging.
            source_coords.append((x, y))

    return samples, json_caption_data, source_coords

#Slides an out_width x screen_height window over the level, keeping only screens that are
#entirely null-free (@ / out-of-bounds), then adds top_pad rows of null padding on top.
#This mirrors the path follower: the screen is real, navigable content (zero out-of-bounds,
#like the path follower's nav window and like the vertical scan), and the only null is the
#synthetic padding added on top. Output scenes are (screen_height + top_pad) tall. Air ('-')
#is legitimate content, so only @ matters. Used by the 'snap' scan mode.
def snap_window_samples(level, tile_to_id, out_width, screen_height, null_chars, top_pad=0, x_stride=1, y_stride=1, direction_captions=False):
    null_id = tile_to_id.get(null_chars[0], 0)
    level_height = len(level)
    level_width = len(level[0])

    samples = []
    json_caption_data = []
    source_coords = []

    #Snap-mode scenes don't come from a walked path, so there's no real "entrance/exit"
    #the way parse_level() has one. We use top_pad as the signal for orientation: wide
    #scenes (top_pad > 0, mirroring the path follower's padding) are treated as moving
    #horizontally so ceiling/floor captions apply the same way path-mode scenes do; tall
    #scenes (top_pad == 0) are treated as moving vertically, so ceiling is correctly
    #skipped for them (ceiling captions only fire for horizontal exit directions).
    if top_pad > 0:
        sample_direction_data = {"entrance_direction": "RIGHT", "exit_direction": "RIGHT"}
    else:
        sample_direction_data = {"entrance_direction": "DOWN", "exit_direction": "DOWN"}

    for y in range(0, level_height - screen_height + 1, y_stride):
        for x in range(0, level_width - out_width + 1, x_stride):
            screen = [level[y+r][x:x+out_width] for r in range(screen_height)]
            if any(c in null_chars for row in screen for c in row):
                continue

            #Synthetic null padding rows on top, then the encoded null-free screen below.
            encoded = [[null_id] * out_width for _ in range(top_pad)]
            for row in screen:
                encoded.append([tile_to_id.get(ch, null_id) for ch in row])
            samples.append(encoded)
            #None (not {}) keeps this parallel with the path-follower's "no captions"
            #convention when direction_captions is off; the caption consumer skips None
            #but KeyErrors on an empty dict.
            json_caption_data.append(sample_direction_data if direction_captions else None)
            #Source coords point at the top-left of the real (null-free) screen, i.e.
            #below the synthetic top_pad rows -- so it points at actual level content.
            source_coords.append((x, y))

    return samples, json_caption_data, source_coords

#Tiles the level into a screens_x x screens_y grid of Mega Man Maker screens (each
#screen_width x screen_height) and slides that grid over the level in (x_stride, y_stride)
#steps. Each screen-sized quadrant counts as "occupied" if it holds at least one non-null
#tile (a real screen, even if it is only walkable '-' sky) and as null if it is entirely
#out-of-bounds void ('@'). A window is kept only when at least min_occupied_quadrants of its
#screens_x*screens_y quadrants are occupied, then top_pad rows of null padding are added on
#top. The content region is (screens_x*screen_width) x (screens_y*screen_height); the final
#scene is top_pad rows taller. With screen 16x14, a 2x2 grid + 4 pad rows yields the 32x32
#scenes used by the 'screen_grid' scan mode. Pass x_stride/y_stride = 16/14 for the standard
#screen-aligned (single-screen-overlap) scan.
def screen_grid_samples(level, tile_to_id, screen_width, screen_height, screens_x, screens_y,
                        top_pad, null_chars, min_occupied_quadrants, x_stride=None, y_stride=None):
    null_set = set(null_chars)
    null_id = tile_to_id.get(null_chars[0], 0)
    content_width = screens_x * screen_width
    content_height = screens_y * screen_height
    x_stride = x_stride or screen_width
    y_stride = y_stride or screen_height
    level_height = len(level)
    level_width = len(level[0])

    samples = []
    json_caption_data = []
    source_coords = []

    for y in range(0, level_height - content_height + 1, y_stride):
        for x in range(0, level_width - content_width + 1, x_stride):
            window = [level[y+r][x:x+content_width] for r in range(content_height)]

            #Count occupied quadrants: each quadrant is one screen_width x screen_height
            #block, occupied if any tile in it is non-null.
            occupied = 0
            for qy in range(screens_y):
                r0 = qy * screen_height
                for qx in range(screens_x):
                    c0 = qx * screen_width
                    if any(window[r0+dr][c0+dc] not in null_set
                           for dr in range(screen_height)
                           for dc in range(screen_width)):
                        occupied += 1
            if occupied < min_occupied_quadrants:
                continue

            #top_pad synthetic null rows on top, then the encoded content window below.
            encoded = [[null_id] * content_width for _ in range(top_pad)]
            for row in window:
                encoded.append([tile_to_id.get(ch, null_id) for ch in row])
            samples.append(encoded)
            #None (not {}) keeps this parallel with the path-follower's "no captions"
            #convention; the caption consumer skips None but KeyErrors on an empty dict.
            json_caption_data.append(None)
            #Source coords point at the top-left of the real content region, i.e. below the
            #synthetic top_pad rows -- so it points at actual level content.
            source_coords.append((x, y))

    return samples, json_caption_data, source_coords

#Parses through one complete level
#width/height are the NAVIGATION (screen) dimensions; out_width/out_height are the
#output scene dimensions (default to a square of side 'width' to match the old behaviour).
def parse_level(tile_to_id, level, width, height, null_chars=['@'], wall_chars=['#'], out_width=None, out_height=None, faithful_vertical=False, start_direction=Direction.RIGHT, print_at_corners=False, change_direction_overrides=[], direction_captions = False):
    level_sample=LevelSample(level, width, height, null_chars, wall_chars, out_width=out_width, out_height=out_height, faithful_vertical=faithful_vertical, start_direction=start_direction, print_at_corners=print_at_corners, change_direction_overrides=change_direction_overrides)

    samples = []
    json_caption_data = []

    #Creates a small json dictionary containin information on if there's a ceiling, bottomless pit, and the entrance/exit directions of the sample
#Parses through one complete level
#width/height are the NAVIGATION (screen) dimensions; out_width/out_height are the
#output scene dimensions (default to a square of side 'width' to match the old behaviour).
def parse_level(tile_to_id, level, width, height, null_chars=['@'], wall_chars=['#'], out_width=None, out_height=None, faithful_vertical=False, start_direction=Direction.RIGHT, print_at_corners=False, change_direction_overrides=[], direction_captions = False):
    level_sample=LevelSample(level, width, height, null_chars, wall_chars, out_width=out_width, out_height=out_height, faithful_vertical=faithful_vertical, start_direction=start_direction, print_at_corners=print_at_corners, change_direction_overrides=change_direction_overrides)

    samples = []
    json_caption_data = []
    source_coords = []

    #Creates a small json dictionary containin information on if there's a ceiling, bottomless pit, and the entrance/exit directions of the sample
    def get_json_caption_data(level_sample: LevelSample, prev_direction, current_direction):
        #Check if there is a wall in each direction blocking us from moving that way
        up_open = Direction.UP.is_possible_to_move_direction(level_sample, check_for_walls=True, check_for_possible=False)
        down_open = Direction.UP.is_possible_to_move_direction(level_sample, check_for_walls=True, check_for_possible=False)
        down_possible = Direction.UP.is_possible_to_move_direction(level_sample, check_for_walls=False, check_for_possible=True)
        bottomless_pit = down_open and not down_possible
        sample_json_data = {
            "entrance_direction": Direction((prev_direction.value+2)%4).name,
            "exit_direction": current_direction.name
        }
        return sample_json_data

    #Direction info for additional output to the json file
    prev_direction = start_direction
    current_direction = prev_direction

    moving=True
    samples.append(level_sample.get_sample_from_idx())
    #Source tile coords (top-left of the navigation window) for this sample, captured at
    #append-time so it lines up with whichever step of the path follower produced it.
    source_coords.append((level_sample.x_idx, level_sample.y_idx))

    #Keep json_caption_data parallel with samples so the zip in main() lines up; append
    #None (not {}) when captions are off, since the caption consumer skips on None but
    #KeyErrors on an empty dict.
    if direction_captions:
        json_caption_data.append(get_json_caption_data(level_sample, prev_direction, current_direction))
    else:
        json_caption_data.append(None)

    while moving:
        prev_direction=current_direction
        moving=level_sample.move_step()
        if not moving: 
            break
        samples.append(level_sample.get_sample_from_idx())
        source_coords.append((level_sample.x_idx, level_sample.y_idx))
        current_direction = level_sample.direction

        if direction_captions:
            json_caption_data.append(get_json_caption_data(level_sample, prev_direction, current_direction))
        else:
            json_caption_data.append(None)

    encoded_samples = []
    for sample in samples:
        encoded_sample = []
        for row in sample:
            encoded_sample.append([tile_to_id[c] for c in row])
        encoded_samples.append(encoded_sample)
    return encoded_samples, json_caption_data, source_coords


#Finds the spawn sample to begin searching
def find_start(level_sample):
    start_y=-1
    start_x=-1

    #Loop through every row to find the spawn location
    for i in range(len(level_sample.level)):
        if level_sample.level[i].find('P')!=-1:
            start_y=i
            start_x=level_sample.level[i].find('P')
            break
    
    if start_y==-1:
        return 0, max(0, len(level_sample.level) - level_sample.height)

    #Continue searching down for the bottom of the level or more null chars
    #We do this to get the full level scene, not just the spawn point and up
    lowest_possible_start = min(len(level_sample.level), start_y+level_sample.height)
    lowest_found = False
    for i in range(start_y, lowest_possible_start):
        if level_sample.level[i][start_x]=='@':
            start_y=i
            start_y=max(0, start_y-level_sample.height)
            lowest_found=True
            break

    #Check to see if we didn't find a lower null char (Meaning we hit the bottom of the level, or the level keeps going down awhile)
    if not lowest_found:
        #Did we reach the bottom of the level?
        if lowest_possible_start==len(level_sample.level):
            start_y=lowest_possible_start
            start_y=max(0, start_y-level_sample.height)
        #If not, the level is vertical downwards, so we need to go up to reach the top
        else:
            #Pretty much the same sequence of checks again, just going up this time, this should only rarely be needed
            highest_possible_start=max(start_y-level_sample.height, 0)
            heighest_found=False
            for i in range(start_y, highest_possible_start, -1):
                if level_sample.level[i][start_x]=='@':
                    start_y=i
                    break
            if not heighest_found:
                start_y=highest_possible_start

    #Start at the left edge if close enough
    if start_x<level_sample.width:
        start_x=0
    else:
        start_x=start_x-level_sample.width
    
    return start_x, start_y


class LevelSample():
    def __init__(self, level, width, height, null_chars=['@'], wall_chars=['#'], out_width=None, out_height=None, faithful_vertical=False, start_direction=Direction.RIGHT, print_at_corners=False, change_direction_overrides=[]):
        self.level=level
        self.width=width
        self.height=height
        self.out_width = out_width if out_width is not None else width
        self.out_height = out_height if out_height is not None else width
        self.faithful_vertical = faithful_vertical
        self.null_chars=null_chars
        self.wall_chars=wall_chars
        self.direction=start_direction
        self.print_at_corners=print_at_corners
        #Built for edge cases, plug in an array of integers to override turning logic, and keep moving forward
        self.change_direction_overrides=change_direction_overrides
        self.move_iter=0 #Tracks what movement we're on for overrides
        self.x_idx, self.y_idx = find_start(self)
    
    #Attempts to move one step forward, returns True if sucessful, False otherwise. Throws an error if it finds a spit path
    def move_step(self):
        if self.check_for_end() and not (self.move_iter in self.change_direction_overrides): 
            return False #We're at the end of the level, so break out
        if self.direction.is_possible_to_move_direction(self, check_for_walls=True) or self.move_iter in self.change_direction_overrides:
            self.direction.move_scene(self) #If the scene ahead is clear, move into it
            return True
        return self.change_direction()
    
    #Changes direction of the sample if it should, prioritizing avoiding null chars
    def change_direction(self):
        if self.print_at_corners:
            self.print_sample()
        _, center, _, left_permeability, _, right_permeability = self.check_travel_movability(check_for_walls=True)

        if left_permeability and right_permeability: #Throw an error if there's a fork in the path
            self.print_sample()
            raise ValueError(f"I don't know where to go! The index is x: {self.x_idx}, y: {self.y_idx}, and I can't decide between {Direction((self.direction.value-1)%4).name} and {Direction((self.direction.value+1)%4).name}. The current move is {self.move_iter}")

        #If either side is accesible to us, we should go that way
        if left_permeability:
            self.direction = Direction((self.direction.value-1)%4)
            self.direction.move_scene(self)
            return True
        if right_permeability:
            self.direction = Direction((self.direction.value+1)%4)
            self.direction.move_scene(self)
            return True

        #All cases are not permeable, so if the center route isn't invalid, we should take it
        if center:
            self.direction.move_scene(self)
            return True
        
        #All other cases should be covered by the check
        raise ValueError(f"We should literally never get here, this is a debugging case. The index is x: {self.x_idx}, y: {self.y_idx}")

    #Checks to see if the end of the level has been reached, returns true if it has
    def check_for_end(self):
        #We care if it's *possible* to move straight, and if the walls to the left and right are closed
        #We're never going to turn into a blocked off wall, more often than not this just leads to errors.
        _, center, _, left, _, right = self.check_travel_movability(check_for_walls=True)
        if not (left or center or right):
            return True #If we can't move any direction except backwards, we're probably at the end of the level
        return False

    #Returns a 6-tuple of the ability to move left, forward, and right (relative to the current direction), the first 3 only check for null, the last 3 check for null and walls
    def check_travel_movability(self, check_for_walls = False):
        direction_left = Direction((self.direction.value-1)%4)
        direction_right = Direction((self.direction.value+1)%4)

        left_possibility = direction_left.is_possible_to_move_direction(self)
        center_possibility = self.direction.is_possible_to_move_direction(self)
        right_possibility = direction_right.is_possible_to_move_direction(self)

        left_permeability = None
        right_permeability = None
        center_permeability = None

        if check_for_walls:
            left_permeability = direction_left.is_possible_to_move_direction(self, check_for_walls=True)
            center_permeability = self.direction.is_possible_to_move_direction(self, check_for_walls=True)
            right_permeability = direction_right.is_possible_to_move_direction(self, check_for_walls=True)
        
        return left_possibility, center_possibility, right_possibility, left_permeability, center_permeability, right_permeability

    #Checks if a sample is out of bounds of the full level, defaulting to the sample
    def is_out_of_bounds(self, x = None, y = None):
        if x is None:
            x = self.x_idx
        if y is None:
            y = self.y_idx
        if (x < 0) or (y < 0) or (x+self.width > len(self.level[0])) or (y+self.height > len(self.level)):
            return True #We are out of bounds
        return False #We are not out of bounds
    
    def print_sample(self):
        sample=self.get_sample_from_idx(pad_sample=False)
        print(f"Level sample at ({self.x_idx}, {self.y_idx}) on move step {self.move_iter}:")
        for row in sample:
            print(row)
        print("\n")

    #Gets a full level sample of the desired size from the top left corner
    def get_sample_from_idx(self, x=None, y=None, pad_sample=True):
        if x is None:
            x = self.x_idx
        if y is None:
            y = self.y_idx

        #Make sure the level sample is in bounds
        if x<0 or y<0:
            raise ValueError(f"X value ({x}) and Y value ({y}) all must be positive.")
        if (y + self.height)>len(self.level) or (x+self.width)>len(self.level[0]):
            raise ValueError(f"This level sample is out of bounds at the bottom or right, with height index {y+self.height}/{len(self.level)} and width index {x+self.width}/{len(self.level[0])}.")

        if not pad_sample:
            return [row[x:x+self.width] for row in self.level[y:y+self.height]]

        null = self.null_chars[0]
        level_height = len(self.level)
        level_width = len(self.level[0])

        col_offset = (self.out_width - self.width) // 2
        row_offset = self.out_height - self.height
        level_x0 = x - col_offset
        level_y0 = y - row_offset

        sample = []
        for r in range(self.out_height):
            ly = level_y0 + r
            above_nav_window = r < row_offset
            out_row = []
            for c in range(self.out_width):
                lx = level_x0 + c
                if above_nav_window and not self.faithful_vertical:
                    out_row.append(null)
                elif 0 <= ly < level_height and 0 <= lx < level_width:
                    out_row.append(self.level[ly][lx])
                else:
                    out_row.append(null)
            sample.append(out_row)

        return sample

if __name__ == "__main__":
    main()
