"""Deterministic captions for Mega Man Maker (MMLV) scenes.

Forked from MM_create_ascii_captions.py, which captions VGLC Mega Man scenes. The two
share most of their logic, but MMLV levels are user-made and far more varied, so the
floor handling here does not assume the floor is the last row of the scene. See
find_floor for what replaced that assumption.
"""

import json
import sys
import os
from collections import namedtuple

from captions.util import extract_tileset, describe_quantity, count_caption_phrase, flood_fill
from Game_MMLV.vglc_to_mmlv import (MULTI_TILE_CHUNKS, SPIKE_ANCHOR_CHARS, SPIKE_CHAR,
                                    find_char_chunks, spike_direction)

import util.common_settings as common_settings




# The last row of the scene (0-indexed). Kept as a fallback and as a default for the
# helpers below; the floor itself is located per scene by find_floor.
FLOOR = common_settings.MEGAMAN_HEIGHT - 1
CEILING = common_settings.MEGAMAN_HEIGHT - 14 # 2

# How far up from the bottom of the scene find_floor looks for the walkable surface.
# A solid row higher than this is a platform or a structure, not the floor.
FLOOR_SEARCH_DEPTH = 4
# How many stacked rows may be absorbed into the floor. Mega Man floors are frequently two
# tiles thick; anything thicker is a solid mass and keeps being described as a cluster.
MAX_FLOOR_THICKNESS = 2

# A vertical opening this tall or shorter is too small for Mega Man to pass through, so it
# does not break a wall. Two rows is a doorway.
MAX_IMPASSABLE_GAP = 1


def _pair(word):
    """Singular/plural pair for a phrase whose plural is just an s."""
    return (word, word + "s")


# Enemy identities, from the decode table in Game_MMLV/mmlv_to_vglc.py. Note that 'a' is both
# Met AND the fallback for every enemy id that table does not recognise, so an 'a' is only
# certainly "the enemy drawn with the Met sprite" -- which is what the scene actually shows,
# and what a caption should therefore say.
ENEMY_NAMES = {
    "a": _pair("met"),
    "<": ("octopus battery", "octopus batteries"),
    "^": ("octopus battery", "octopus batteries"),   # same enemy, vertical variant
    "c": _pair("beak"),
    "d": ("picketman", "picketmen"),
    "e": _pair("screw bomber"),
    "f": ("big eye", "big eyes"),
    "g": _pair("spine"),
    "h": ("crazy razy", "crazy razies"),
    "i": _pair("watcher"),
    "I": _pair("tackle fire"),
    "j": _pair("killer bullet"),
    "k": _pair("killer bullet spawner"),
    "n": _pair("mambu"),
    "o": _pair("mambu spawner"),
    "p": _pair("footholder"),
    "r": _pair("sniper joe"),
    "S": ("boss", "bosses"),
}

# Power-up and hazard identities, from Game_MMLV/MMLV.json and util/descriptive_tilesets.py.
# Like the enemies, each is counted per tile under its own name instead of pooled into one
# "powerups" or "hazards" phrase, the way the Mario Maker 2 captioner names every tile.
POWERUP_NAMES = {
    "W": ("large weapon energy", "large weapon energies"),
    "w": ("small weapon energy", "small weapon energies"),
    "L": ("large life energy", "large life energies"),
    "l": ("small life energy", "small life energies"),
    "+": _pair("1-up"),
    "*": _pair("yashichi"),
}

HAZARD_NAMES = {
    "C": _pair("hazard block"),
    "G": _pair("right fire emitter"),
    "J": _pair("left fire emitter"),
}

# Lava is placed tile by tile, so each connected body of it is one pool, described by its
# length instead of counted per tile: "one long lava pool at left." Lava floods down from its
# surface (see flood_liquids_down in Game_MMLV/mmlv_to_vglc.py), so a pool's depth is
# incidental and it is measured by width.
LAVA_CHAR = "!"
LAVA_NAMES = _pair("lava pool")

# A lava pool or spike pit up to this wide is short, a wider one long, and one spanning the
# whole scene full. Splits spike pits in 16x16 scenes about 60% short, 35% long, 5% full; no
# downloaded level has lava yet, so pools borrow the same split.
SHORT_STRIP_MAX = 3

# Spikes point away from the block they sit on (spike_direction), which separates what they
# are part of. Spikes pointing up either line a pit, boxed in by solid blocks at both ends, or
# sit loose on the floor. A spike pit is one object per connected strip, described by its
# width like a lava pool; every other spike is counted tile by tile, by the surface it is on.
SPIKE_NAMES = {
    "pit": _pair("spike pit"),
    "floor": _pair("floor spike"),
    "ceiling": _pair("ceiling spike"),
    "wall": _pair("wall spike"),
}

# Tile types that carry no phrase of their own in the Mega Man captioner, each counted and
# named on its own.
#
# Conveyors and breakable blocks are solid, so they also take part in the floor, platform and
# cluster logic further down. The count here is additional information about the same tiles,
# not a replacement, in the same way that a footholder is both an enemy and a moving platform.
# `multi_tile` marks the types that occupy more than one tile in Mega Man Maker, per the
# footprint tables in Game_MMLV/mmlv_to_vglc.py: a boss door is 2x4 or 4x2, a key door 1x3 or
# 3x1, a teleporter 2x2, and conveyors and platform tracks run to any length. Those are counted
# as objects by find_objects, so one 2x4 boss door reads as "one boss door" rather than
# "several doors", and two teleporters placed side by side still read as two. The genuinely 1x1
# types are counted tile by tile.
TileType = namedtuple("TileType", "chars names multi_tile")

MMLV_TILE_TYPES = (
    TileType(">",   _pair("right conveyor"),  True),
    TileType("E",   _pair("left conveyor"),   True),
    TileType("VY",  _pair("key door"),        True),
    TileType("D",   _pair("boss door"),       True),
    TileType("T",   _pair("teleporter"),      True),
    TileType("=",   _pair("platform path"),   True),
    TileType("K",   _pair("key"),             False),
    TileType("X",   _pair("checkpoint"),      False),
    TileType("s",   _pair("spring"),          False),
    TileType("x",   _pair("fan"),             False),
    TileType("Z",   _pair("level exit"),      False),
    TileType("P",   _pair("spawn point"),     False),
    TileType("t",   _pair("fake block"),      False),
)

# Moving platforms are found as horizontal runs rather than counted as tiles, so they are
# labelled separately, each run named after the tile it is built from.
MOVING_PLATFORM_NAMES = {
    "F": _pair("falling platform"),
    "M": _pair("moving platform"),
    "R": _pair("rising platform"),
    "p": _pair("footholder platform"),
}

# The biggest single breakable block is 2x2, so a cluster of more cells than this has to be
# several blocks: a thick wall or mass to dig through rather than one obstacle. Those clusters
# are called large; the rest keep the plain cluster phrase.
LARGE_CLUSTER_CELLS = 4

# A moving platform sits on one node of its own track, and that node decodes to the platform
# 'M' instead of '=', so 'M' joins the track on either side of it rather than splitting it.
TRACK_CHAR = "="
TRACK_CONNECTORS = "M"

# This is used for describing locations, but it doesn't work well
STANDARD_WIDTH = common_settings.MEGAMAN_WIDTH

LEFT = STANDARD_WIDTH / 3
RIGHT = STANDARD_WIDTH - LEFT

# Something spanning more than this share of the scene's width is everywhere rather than in
# one third of it, so it reads "across" instead of being placed by its midpoint, which put a
# full-width platform "at center".
ACROSS_FRACTION = 2 / 3


# Could define these via the command line, but for now they are hardcoded
coarse_locations = True
coarse_counts = True
pluralize = True
give_staircase_lengths = False


def describe_location(x):
    """
        Describes where a column sits across the scene: "left", "center" or "right".

        Only the horizontal axis is described. A vertical word would have to be measured
        against the floor, which moves from scene to scene, and it multiplies the number of
        phrases a single feature type can produce -- a scene with platforms at eight
        positions was getting eight separate platform phrases.
    """
    if x < LEFT:
        return "left"
    elif x < RIGHT:
        return "center"
    return "right"


def describe_span(start_x, end_x):
    """The location phrase for something covering columns start_x to end_x: "across" when it
    spans most of the scene's width (ACROSS_FRACTION)"""
    if end_x - start_x + 1 > ACROSS_FRACTION * STANDARD_WIDTH:
        return "across"
    return "at " + describe_location((start_x + end_x) / 2.0)


def find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor, min_run_length=2, require_above_below_not_solid=False, exclude_rows = [], already_accounted = set()):
    """
    Finds horizontal lines (runs) of tiles with the target descriptor.
    - Skips the FLOOR row
    - Can require non-solid space above and below (for platforms)
    - Skips tiles in already_accounted, so a moving platform or key door that is solid isn't
      also found as a plain platform; exclude_rows may not be needed because of it
    Returns a list of (y, start_x, end_x) tuples
    """
    lines = []
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0

    #print((10,0) in already_accounted)

    for y in range(height):  # Skip FLOOR row
        
        if y in exclude_rows:
            continue # Could skip ceiling

        x = 0
        while x < width:
            tile_char = id_to_char[scene[y][x]]
            descriptors = tile_descriptors.get(tile_char, [])

            if target_descriptor not in descriptors or (y, x) in already_accounted:
                x += 1
                continue

            # If required, check for passable tiles above and below
            if require_above_below_not_solid:
                # Above
                if y > 0:
                    above_char = id_to_char[scene[y - 1][x]]
                    if "solid" in tile_descriptors.get(above_char, []):
                        x += 1
                        continue
                else:
                    x += 1
                    continue
                # Below
                if y + 1 < height:
                    below_char = id_to_char[scene[y + 1][x]]
                    if "solid" in tile_descriptors.get(below_char, []):
                        x += 1
                        continue
                else:
                    x += 1
                    continue

            # Start of valid run
            possible_locations = set()
            run_start = x
            while x < width:
                tile_char = id_to_char[scene[y][x]]
                descriptors = tile_descriptors.get(tile_char, [])

                if target_descriptor in descriptors and (y, x) not in already_accounted:
                    if require_above_below_not_solid:
                        if y > 0 and "solid" in tile_descriptors.get(id_to_char[scene[y - 1][x]], []):
                            break
                        if y + 1 < height and "solid" in tile_descriptors.get(id_to_char[scene[y + 1][x]], []):
                            break

                    possible_locations.add( (y,x) )
                    x += 1
                else:
                    break
            run_length = x - run_start
            if run_length >= min_run_length:
                already_accounted.update(possible_locations) # Blocks of the line are now accounted for
                lines.append((y, run_start, x - 1))

    return lines

    


def describe_horizontal_lines(lines, label, describe_locations, describe_absence):
    if not lines:
        if describe_absence:
            return f" no {label}s."
        else:
            return ""
        
    if describe_locations:
        
        if coarse_locations:
            location_counts = {}
            for y, start_x, end_x in sorted(lines):
                location_str = describe_span(start_x, end_x)
                if location_str in location_counts:
                    location_counts[location_str] += 1
                else:
                    location_counts[location_str] = 1

            return " " + ". ".join([f"{describe_quantity(count) if coarse_counts else count} {label}{'s' if pluralize and count > 1 else ''} {location}" for location, count in location_counts.items()]) + "."
            
        else:
            parts = []
            for y, start_x, end_x in sorted(lines):
                parts.append(f"{y} (cols {start_x}-{end_x})")
            # Fix unbound variable 'count'
            count = len(lines)
            location_description = f"at row{'s' if pluralize and count > 1 else ''} " + ", ".join(parts)
        
            plural = label + "s" if pluralize and count > 1 else label
            return f" {describe_quantity(count) if coarse_counts else count} {plural} " + location_description + "."

    else: # Not describing locations at all
        count = len(lines)
        return f" {describe_quantity(count) if coarse_counts else count} {label}{'s' if pluralize and count != 1 else ''}."


def find_char_structures(scene, id_to_char, chars, already_accounted, diagonal=False):
    """Connected groups of tiles drawn from `chars`, as a list of sets of (row, col).

    find_solid_structures flood-fills by descriptor, which would pull key doors in with the
    breakable blocks and breakable blocks in with the terrain. This fills over an explicit
    set of characters instead, so a family of tiles can be recognised as its own structures
    while still being made of solid tiles. `diagonal` also joins tiles that only touch at a
    corner.
    """
    chars = set(chars)
    targets = {(r, c) for r, row in enumerate(scene) for c, t in enumerate(row)
               if id_to_char[t] in chars and (r, c) not in already_accounted}
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    structures = []
    # Walk the tiles in reading order so the phrases come out left to right rather than in
    # whatever order the set happens to yield.
    for start in sorted(targets, key=lambda pos: (pos[1], pos[0])):
        if start not in targets:
            continue
        targets.discard(start)
        stack = [start]
        structure = {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                neighbour = (r + dr, c + dc)
                if neighbour in targets:
                    targets.discard(neighbour)
                    structure.add(neighbour)
                    stack.append(neighbour)
        structures.append(structure)
    return structures


def find_footprint_objects(scene, id_to_char, ch, already_accounted):
    """One set of (row, col) per placed object of a fixed-footprint tile: teleporter, boss door,
    key door.

    Flood fill would merge two teleporters placed side by side into one, so the cells are cut
    into the footprints the reverse converter rebuilds objects from (MULTI_TILE_CHUNKS), trying
    the shapes in its order. Cells left over belong to an object cropped by the scene edge or
    partly drawn over by another tile; each connected piece of those still counts as one object.
    """
    rows = ["".join(id_to_char[t] for t in row) for row in scene]
    claimed = set(already_accounted)
    objects = []
    for shape in MULTI_TILE_CHUNKS[ch]:
        anchor_c, anchor_r = shape["anchor"]
        anchors, members = find_char_chunks(rows, ch, shape["cells"], shape["anchor"], exclude=claimed)
        for r, c in anchors:
            top, left = r - anchor_r, c - anchor_c
            objects.append({(top + dr, left + dc) for dc, dr in shape["cells"]})
        claimed |= members
    return objects + find_char_structures(scene, id_to_char, ch, claimed)


def find_tracks(scene, id_to_char, already_accounted):
    """One set of (row, col) per moving platform track, holding just its '=' cells.

    A track can step diagonally, so corner contact keeps it in one piece, and the platform
    riding it bridges the node it sits on (see TRACK_CONNECTORS).
    """
    pieces = find_char_structures(scene, id_to_char, TRACK_CHAR + TRACK_CONNECTORS,
                                  already_accounted, diagonal=True)
    tracks = [{(r, c) for r, c in piece if id_to_char[scene[r][c]] == TRACK_CHAR} for piece in pieces]
    return [track for track in tracks if track]


def find_spike_groups(scene, id_to_char, already_accounted):
    """The scene's spikes as (kind, cells) pairs, kind being a SPIKE_NAMES key. A spike pit is
    one pair per connected strip; every other spike is a pair of its own.

    Spikes are split by the way they point first, so a pit is only ever made of upward spikes.
    An upward strip is a pit when both of its ends are boxed in, by a solid block level with it
    or just above it, or by the edge of the scene: it is sunk into or flush with the floor
    rather than sitting on it.
    """
    rows = ["".join(id_to_char[t] for t in row) for row in scene]
    width = len(rows[0]) if rows else 0
    # Beyond the edge of the scene counts as solid: a spike there is anchored to a block the
    # crop cut off, like a column of wall spikes whose wall is in the next scene over.
    edge = "#"
    padded = [edge * (width + 2)] + [edge + row + edge for row in rows] + [edge * (width + 2)]
    by_kind = {"up": set(), "wall": set(), "ceiling": set()}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch != SPIKE_CHAR or (r, c) in already_accounted:
                continue
            direction = spike_direction(padded, r + 1, c + 1, SPIKE_ANCHOR_CHARS)
            kind = {"up": "up", "down": "ceiling"}.get(direction, "wall")
            by_kind[kind].add((r, c))

    # The spike at either end of a column of wall spikes can rest on the floor or hang from the
    # ceiling, which points it up or down, but it is still on the wall. A lone upward or
    # downward spike directly above or below a wall spike, with no spike of its own kind beside
    # it, counts as a wall spike.
    changed = True
    while changed:
        changed = False
        for kind in ("up", "ceiling"):
            for r, c in sorted(by_kind[kind]):
                beside = (r, c - 1) in by_kind[kind] or (r, c + 1) in by_kind[kind]
                capping = (r - 1, c) in by_kind["wall"] or (r + 1, c) in by_kind["wall"]
                if capping and not beside:
                    by_kind[kind].discard((r, c))
                    by_kind["wall"].add((r, c))
                    changed = True

    def strips(cells):
        # Flood fill only within `cells`, so the other directions' spikes stay out.
        others = {(r, c) for r, row in enumerate(rows) for c, _ in enumerate(row)} - cells
        return find_char_structures(scene, id_to_char, SPIKE_CHAR, others)

    def boxed_in(strip_rows, col):
        if col < 0 or col >= width:
            return True
        return any(0 <= r < len(rows) and rows[r][col] in SPIKE_ANCHOR_CHARS
                   for r in strip_rows | {min(strip_rows) - 1})

    groups = []
    for strip in strips(by_kind["up"]):
        strip_rows = {r for r, _ in strip}
        cols = [c for _, c in strip]
        if boxed_in(strip_rows, min(cols) - 1) and boxed_in(strip_rows, max(cols) + 1):
            groups.append(("pit", strip))
        else:
            groups += [("floor", {cell}) for cell in sorted(strip)]
    for kind in ("ceiling", "wall"):
        groups += [(kind, {cell}) for cell in sorted(by_kind[kind])]
    return groups


def find_objects(scene, id_to_char, chars, already_accounted):
    """One set of (row, col) per placed object of a multi-tile type, for each char in `chars`.

    Each char is grouped on its own, so a vertical key door touching a horizontal one is still
    two key doors, though both count under the same name.
    """
    objects = []
    for ch in chars:
        if ch in MULTI_TILE_CHUNKS:
            objects += find_footprint_objects(scene, id_to_char, ch, already_accounted)
        elif ch == TRACK_CHAR:
            objects += find_tracks(scene, id_to_char, already_accounted)
        elif ch == SPIKE_CHAR:
            objects += [cells for _kind, cells in find_spike_groups(scene, id_to_char, already_accounted)]
        else:
            objects += find_char_structures(scene, id_to_char, ch, already_accounted)
    # Left to right, as find_char_structures orders them, so the phrases keep that order.
    return sorted(objects, key=lambda cells: min((c, r) for r, c in cells))


def describe_structure_counts(structures, names, describe_locations=False):
    """One phrase per group of structures, each structure counting as a single object.

    Used for the tile types that span several tiles, so that a 2x4 boss door is one door.
    """
    if not structures:
        return []
    singular, plural = names

    def phrase(items, where=None):
        count = len(items)
        label = plural if pluralize and count > 1 else singular
        text = f" {describe_quantity(count) if coarse_counts else count} {label}"
        text += f" {where}." if where else "."
        return (text, set().union(*items))

    if describe_locations and coarse_locations:
        by_location = {}
        for struct in structures:
            cols = [c for _, c in struct]
            where = describe_span(min(cols), max(cols))
            by_location.setdefault(where, []).append(struct)
        return [phrase(group, where) for where, group in by_location.items()]

    return [phrase(structures)]


def describe_tile_structures(structures, singular, plural, describe_locations=False,
                             describe_absence=False):
    """Phrases for groups of one tile family: a lone tile keeps its own name, a group of two
    or more becomes a cluster of them, and a cluster bigger than LARGE_CLUSTER_CELLS a large
    cluster. Returns (phrase, coordinates) tuples.
    """
    singles = [s for s in structures if len(s) == 1]
    clusters = [s for s in structures if 1 < len(s) <= LARGE_CLUSTER_CELLS]
    large = [s for s in structures if len(s) > LARGE_CLUSTER_CELLS]
    result = []

    for items, (one, many) in ((singles, (singular, plural)),
                               (clusters, (singular + " cluster", singular + " clusters")),
                               (large, ("large " + singular + " cluster", "large " + singular + " clusters"))):
        if not items:
            continue
        if describe_locations and coarse_locations:
            # One phrase per location, so "two breakable blocks at left bottom" stays true.
            by_location = {}
            for struct in items:
                cols = [c for _, c in struct]
                where = describe_span(min(cols), max(cols))
                by_location.setdefault(where, []).append(struct)
            for where, group in by_location.items():
                count = len(group)
                label = many if pluralize and count > 1 else one
                coords = set().union(*group)
                result.append((f" {describe_quantity(count) if coarse_counts else count} {label} {where}.", coords))
        else:
            count = len(items)
            label = many if pluralize and count > 1 else one
            coords = set().union(*items)
            result.append((f" {describe_quantity(count) if coarse_counts else count} {label}.", coords))

    if not result and describe_absence:
        result.append((f" no {plural}.", set()))
    return result


def find_solid_structures(scene, id_to_char, tile_descriptors, already_accounted):
    """Find unaccounted solid block structures"""
    visited = set()
    structures = []

    for row in range(len(scene)):
        for col in range(len(scene[0])):
            if (row, col) in visited or (row, col) in already_accounted:
                continue
            tile = scene[row][col]
            descriptors = tile_descriptors.get(id_to_char[tile], [])
            if "solid" in descriptors:
                structure = flood_fill(scene, visited, row, col, id_to_char, tile_descriptors, already_accounted)
                if len(structure) >= 3:  # Ignore tiny groups of blocks
                    structures.append(structure)
                    already_accounted.update(structure)

    return structures


def describe_structures(structures, ceiling_row=CEILING, floor_row=FLOOR, describe_absence=False, describe_locations=False, debug=False):
    """
        Returns a list of tuples (phrase, coordinates) where coordinates is a set of (row, col) positions
        associated with the phrase describing the structures of that type.
    """
    # Map each description to its list of structures
    desc_to_structs = {}
    
    for struct in structures:
        min_row = min(pos[0] for pos in struct)
        max_row = max(pos[0] for pos in struct)
        min_col = min(pos[1] for pos in struct)
        max_col = max(pos[1] for pos in struct)

        width = max_col - min_col + 1
        height = max_row - min_row + 1

        attached_to_ceiling = any(r == ceiling_row for r, c in struct)
        in_contact_with_floor = any(r == floor_row - 1 for r, c in struct)


        if not attached_to_ceiling and width <= 2 and height >= 3 and in_contact_with_floor:
            desc = "tower"
        elif all((r, c) in struct for r in range(min_row, max_row + 1) for c in range(min_col, max_col + 1)):
            desc = "rectangular block cluster"
        #elif not attached_to_ceiling and width >= 3 and height <= 2 and in_contact_with_floor:
        #    desc = "wall"
        else:
            desc = "irregular block cluster"

        if debug:
            print(f"{desc} at {min_row} {max_row} {min_col} {max_col}: {struct}: attached_to_ceiling: {attached_to_ceiling}, in_contact_with_floor: {in_contact_with_floor}")

        if describe_locations:
            if coarse_locations:
                desc += " " + describe_span(min_col, max_col)
            else:
                desc += f" from row {min_row} to {max_row}, columns {min_col} to {max_col}"

        # Group structures by their description
        if desc not in desc_to_structs:
            desc_to_structs[desc] = []
        desc_to_structs[desc].append(struct)

    # Prepare phrases with their associated coordinates
    result = []
    
    # Process existing structures
    for desc, struct_list in desc_to_structs.items():
        count = len(struct_list)
        # Combine all coordinates for this description type
        all_coords = set()
        for struct in struct_list:
            all_coords.update(struct)
            
        if count == 1:
            # Need space in front
            phrase = f" one {desc}"
        else:
            # Pluralize the first word
            words = desc.split()
            for i in range(len(words)):
                if words[i] == "tower":
                    words[i] = "towers"
                #elif words[i] == "wall":
                #    words[i] = "walls"
                elif words[i] == "cluster":
                    words[i] = "clusters"
            phrase = f" {describe_quantity(count)} " + " ".join(words)
        
        result.append((phrase + ".", all_coords))

    # Handle absence descriptions if needed
    if describe_absence:
        absent_types = {"tower": set(), "rectangular block cluster": set(), "irregular block cluster": set()}
        described_types = desc_to_structs.keys()
        
        for absent_type in absent_types:
            if absent_type not in described_types:
                result.append((f" no {absent_type}s.", set()))

    return result if result else []


def find_ladders(scene, ladder_ids, already_accounted=set(), describe_absence=False, floor_row=None):
    """
    Finds vertical runs of ladder tiles and classifies each by where it connects
    within the playable area (excluding the 2-row ceiling and the floor row):
    top only, bottom only, both (full height), or neither (middle).

    `floor_row` is the walkable surface found by find_floor. A ladder that reaches it
    connects to the bottom even when the scene continues below the floor.
    """
    ladders = []  # list of (start_y, end_y, x)
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0

    ceiling_row = 2          # rows 0-1 are non-playable ceiling
    if floor_row is None:
        floor_row = height - 1
    playable_top = ceiling_row
    playable_bottom = floor_row 

    for x in range(width):
        y = 0
        while y < height:
            if scene[y][x] not in ladder_ids:
                y += 1
                continue

            possible_locations = set()
            run_start = y
            while y < height:
                if scene[y][x] in ladder_ids:
                    possible_locations.add((y, x))
                    y += 1
                else:
                    break
            already_accounted.update(possible_locations)
            run_end = y - 1
            ladders.append((run_start, run_end, x))

    if not ladders:
        if describe_absence:
            return [(" no ladders.", set())]
        else:
            return []

    categories = {"both": [], "top": [], "bottom": [], "middle": []}
    for start_y, end_y, x in ladders:
        connects_top = start_y <= playable_top
        connects_bottom = end_y >= playable_bottom
        if connects_top and connects_bottom:
            categories["both"].append((start_y, end_y, x))
        elif connects_top:
            categories["top"].append((start_y, end_y, x))
        elif connects_bottom:
            categories["bottom"].append((start_y, end_y, x))
        else:
            categories["middle"].append((start_y, end_y, x))

    suffix_map = {"top": "at top", "bottom": "at bottom", "middle": "in the middle"}

    # One (phrase, coords) tuple per ladder category, so each subtype gets its own
    # entry in `details` instead of being merged into one combo string.
    result = []
    for cat in ("both", "top", "bottom", "middle"):
        items = categories[cat]
        if not items:
            continue
        count = len(items)
        plural = "s" if pluralize and count != 1 else ""
        quantity = describe_quantity(count) if coarse_counts else count

        if cat == "both":
            phrase = f" {quantity} full height ladder{plural}."
        else:
            phrase = f" {quantity} ladder{plural} {suffix_map[cat]}."

        coords = {(y, x) for start_y, end_y, x in items for y in range(start_y, end_y + 1)}
        result.append((phrase, coords))

    return result

def find_water_caption(scene, empty_ids, water_ids, describe_absence=False):
    """
        Finds the ratio of water to all empty tiles, and returns the caption for it
    """
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0

    empty_count = 0
    water_count = 0

    for x in range(width):
        for y in range(height):
            id_at_loc = scene[y][x] # Get char at location

            if id_at_loc in empty_ids:
                empty_count += 1
                if id_at_loc in water_ids: #Water tiles should always be empty tiles as well, so we nest them to prevent errors
                    water_count += 1
    
    if empty_count==0 or water_count==0: #Need an escape so we don't devide by 0
        if describe_absence:
            return " no water."
        else:
            return ""
    
    ratio = water_count/empty_count
    
    if ratio < 0.35:
        return " some water."
    elif ratio >= .35 and ratio < .65:
        return " half water."
    elif ratio >= .65 and ratio != 1.0:
        return " mostly water."
    elif ratio == 1.0:
        return " all water."
    
    raise ValueError(f"It shouldn't be possible to get here. Error in describing water with air/water ratio of {ratio}")
    

# We need a seperate function so we avoid counting things like spikes as the ceiling
def analyze_ceiling(scene, wall_ids, describe_absence, ceiling_row = 2):
    """
    Analyzes ceiling row (0-based index) to detect a ceiling.
    Returns a caption phrase or an empty string if no ceiling is detected.
    """
    WIDTH = len(scene[0])

    row = scene[ceiling_row]
    #Count the number of solid tiles in the ceiling row
    solid_count = sum(1 for tile in row if tile in wall_ids)
    
    if solid_count == WIDTH:
        return " full ceiling."
    elif solid_count > WIDTH//2:
        # Count contiguous gaps of passable tiles
        gaps = 0
        in_gap = False
        for tile in row:
            # Get gaps if the tile at a point isn't a solid block
            if tile not in wall_ids:
                if not in_gap:
                    gaps += 1
                    in_gap = True
            else:
                in_gap = False
        result = f" ceiling with {describe_quantity(gaps) if coarse_counts else gaps} gap" + ("s" if pluralize and gaps != 1 else "") + "."

        # Adding the "moving" check should make this code unnecessary
        #if result == ' ceiling with no gaps.':
        #    print("This should not happen: ceiling with no gaps")
        #    print("ceiling_row:", scene[ceiling_row])
        #    result = " full ceiling."

        return result
    elif describe_absence:
        return " no ceiling."
    else:
        return ""  # Not enough solid tiles for a ceiling

# The floor of an MMLV scene is not reliably the last row: plenty of user-made levels draw
# the ground one row up, sink a pit below it, or stack it two tiles thick. Everything from
# here to describe_floor_occupancy exists to find the floor instead of assuming it.

Floor = namedtuple("Floor", "row rows occupancy accounted surface")
"""row: the walkable surface row (top of the floor), used wherever the rest of the module
   needs to know where the playable area ends.
rows: every row belonging to the floor, including the wiggle row below it.
occupancy: per column, True when that column is standable somewhere in the floor.
accounted: tiles claimed by the floor, empty cells included, so that gaps in the floor
   cannot resurface later as platforms, clusters or loose blocks.
surface: just the solid tiles, for highlighting the phrase in the data browser."""


def count_runs(flags, value):
    """Number of maximal contiguous runs of `value` in `flags`."""
    runs = 0
    in_run = False
    for flag in flags:
        if flag == value:
            if not in_run:
                runs += 1
                in_run = True
        else:
            in_run = False
    return runs


def row_is_floor_like(scene, row, non_gap_ids):
    """True when more of the row's columns are standable than not.

    This is the same majority test the original bottom-row-only analyze_floor used to
    decide between "floor with N gaps" and "giant gap with N chunks of floor", reused here
    so the module has one notion of what counts as a floor rather than two thresholds.
    """
    width = len(scene[0])
    solid_count = sum(1 for tile in scene[row] if tile in non_gap_ids)
    return solid_count * 2 > width


def find_floor(scene, non_gap_ids, search_depth=FLOOR_SEARCH_DEPTH, max_thickness=MAX_FLOOR_THICKNESS):
    """Locates the floor. Returns a Floor, or None when the scene has no floor to find.

    Three things happen, in order:

    1. The base is the LOWEST row within `search_depth` of the bottom that qualifies as a
       floor. A scene whose ground sits a row or two above an empty bottom row is then
       described as a floor rather than as a giant gap with one chunk of floor.
    2. The floor grows upward through its own thickness: a row is absorbed when it is
       floor-like and every one of its solid tiles rests directly on the row below, so the
       top half of a two-thick floor stops being reported as a rectangular block cluster.
       `max_thickness` keeps this from swallowing a tall solid mass.
    3. A column counts as standable if it is solid anywhere in the floor or in the single
       row below it (the "wiggle" row), so a surface that dips by one row reads as one
       floor instead of a gap plus a loose block.

    Whatever sits underneath a standable column is buried: Mega Man can never reach it, so
    it belongs to the floor rather than being described as its own cluster or as loose
    blocks. Tiles under a GAP in the floor are reachable by falling in, so those are left
    for the rest of the caption to describe.

    Returning None means no row near the bottom is standable at all, i.e. a genuine pit
    room; the caller falls back to describing the last row exactly as before.
    """
    height = len(scene)
    width = len(scene[0]) if height else 0
    if not height or not width:
        return None

    base = None
    for row in range(height - 1, max(-1, height - 1 - search_depth), -1):
        if row_is_floor_like(scene, row, non_gap_ids):
            base = row
            break
    if base is None:
        return None

    top = base
    while (base - top + 1) < max_thickness and top > 0:
        above = top - 1
        if not row_is_floor_like(scene, above, non_gap_ids):
            break
        # An unsupported solid tile has air beneath it, which makes the row an overhang or
        # a separate structure rather than more of the same floor.
        supported = all(
            scene[above + 1][c] in non_gap_ids
            for c in range(width) if scene[above][c] in non_gap_ids
        )
        if not supported:
            break
        top = above

    band_rows = list(range(top, base + 1))
    wiggle_row = base + 1 if base + 1 < height else None

    support_row = {}
    for c in range(width):
        # The topmost solid tile of the column is the one that gets stood on.
        support = next((r for r in band_rows if scene[r][c] in non_gap_ids), None)
        if support is None and wiggle_row is not None and scene[wiggle_row][c] in non_gap_ids:
            support = wiggle_row
        if support is not None:
            support_row[c] = support

    occupancy = [c in support_row for c in range(width)]
    surface = {(r, c) for c, r in support_row.items()}

    # The whole band, empty cells included, plus everything buried under a standable column.
    accounted = {(r, c) for r in band_rows for c in range(width)}
    for c, support in support_row.items():
        accounted.update((r, c) for r in range(support, height))

    rows = list(band_rows)
    if wiggle_row is not None:
        rows.append(wiggle_row)

    return Floor(row=top, rows=rows, occupancy=occupancy, accounted=accounted, surface=surface)


def describe_floor_occupancy(occupancy, describe_absence):
    """Turns a column-wise standable/not view of the floor into a caption phrase.

    The vocabulary is unchanged from the original bottom-row-only analyze_floor, so scenes
    whose floor really is the last row keep their existing captions.
    """
    width = len(occupancy)
    solid_count = sum(1 for standable in occupancy if standable)
    passable_count = width - solid_count

    if solid_count == width:
        return " full floor."
    elif solid_count == 0:
        if describe_absence:
            return " no floor."
        else:
            return ""
    elif solid_count > passable_count:
        gaps = count_runs(occupancy, False)
        return f" floor with {describe_quantity(gaps) if coarse_counts else gaps} gap" + ("s." if pluralize and gaps != 1 else ".")
    else:
        chunks = count_runs(occupancy, True)
        return f" giant gap with {describe_quantity(chunks) if coarse_counts else chunks} chunk"+("s" if pluralize and chunks != 1 else "")+" of floor."


def analyze_floor(scene, wall_ids, describe_absence, floor_row=FLOOR, ladder_ids=None):
    """Describes a single row of the scene as the floor.

    This is the fallback for scenes where find_floor comes up empty, and it behaves exactly
    like the Mega Man version: a ladder tile is not a fall-through gap, so it counts as
    floor, but anything else passable does not.
    """
    non_gap_ids = set(wall_ids) | set(ladder_ids or [])
    occupancy = [tile in non_gap_ids for tile in scene[floor_row]]
    return describe_floor_occupancy(occupancy, describe_absence)


def generate_captions(dataset_path, tileset_path, output_path, describe_locations=True, describe_absence=False,
                      caption_mode="legacy", caption_key="deterministic_captions"):
    """Processes the dataset and generates captions for each level scene."""
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    save_level_data(dataset, tileset_path, output_path, describe_locations, describe_absence,
                    caption_mode=caption_mode, caption_key=caption_key)
    print(f"Captioned dataset saved to {output_path}")

def save_level_data(dataset, tileset_path, output_path, describe_locations=True, describe_absence=False,
                    caption_mode="legacy", caption_key="deterministic_captions"):
    """Add a deterministic caption to every scene.

    "keyed" stores it as a one-element list under caption_key (default
    "deterministic_captions"), so a scene can carry captions from several sources at once;
    "legacy" stores it in the single "caption" string. Either way every other input attribute
    is copied through, so a dataset that already carries metadata or LLM captions accumulates
    sources rather than being replaced.
    """
    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)

    # Generate captions
    captioned_dataset = []
    for i, combined_scene in enumerate(dataset):
        # Blank for Mega Man
        is_dict = isinstance(combined_scene, dict)
        if is_dict:
            scene = combined_scene['scene']
            data = combined_scene.get('data', None)
        else:
            scene = combined_scene
            data = None
        caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors,
                                 describe_locations, describe_absence, data)

        # Copy all input attributes (metadata + captions from other sources) so they carry
        # through; only the scene/caption fields below are (re)written.
        entry = dict(combined_scene) if is_dict else {}
        entry["scene"] = scene
        if caption_mode == "keyed":
            entry[caption_key] = [caption]
        else:
            entry["caption"] = caption
        captioned_dataset.append(entry)

    # Save new dataset with captions
    with open(output_path, "w") as f:
        json.dump(captioned_dataset, f, indent=4)

def detect_edge_walls(scene, wall_ids, ceiling_row=2, floor_row=15):
    """
    Detects 'left wall', 'perforated left wall', 'right wall', or 'perforated right wall'.
    
    1. Solid Wall: A SINGLE flood-filled component of solid blocks within the 
       3 outermost columns spans continuously from `ceiling_row` to `floor_row`. The top of
       the wall may fall short of `ceiling_row` by up to MAX_IMPASSABLE_GAP rows: a slab
       that stops a row below the ceiling is still a solid wall, not a perforated one,
       because nothing can get over it.
    2. Perforated Wall: No single flood-filled shape spans top-to-bottom, but 
       there are NO contiguous vertical gaps >= 2 tiles tall across the 3 boundary columns.
    3. None: Any vertical gap >= 2 tiles tall exists across all 3 columns.
    """
    height = len(scene)
    width = len(scene[0]) if height > 0 else 0
    wall_set = set(wall_ids)

    def evaluate_side(col_indices):
        valid_cols = set(col_indices)
        
        # Collect all solid blocks in the 3-column boundary strip within playable rows
        solid_nodes = {
            (r, c) for r in range(ceiling_row, floor_row + 1)
            for c in col_indices if scene[r][c] in wall_set
        }

        # --- Pass 1: Check for a SINGLE Flood-Filled Shape from Top to Bottom ---
        visited = set()
        for start_node in solid_nodes:
            if start_node in visited:
                continue

            # BFS / Flood Fill for a SINGLE component
            queue = [start_node]
            component = {start_node}
            visited.add(start_node)
            
            reaches_top = False
            reaches_bottom = False

            while queue:
                curr_r, curr_c = queue.pop(0)
                if curr_r <= ceiling_row + MAX_IMPASSABLE_GAP:
                    reaches_top = True
                if curr_r == floor_row:
                    reaches_bottom = True

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    neighbor = (nr, nc)
                    if (ceiling_row <= nr <= floor_row and 
                        nc in valid_cols and 
                        neighbor in solid_nodes and 
                        neighbor not in visited):
                        
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)

            # If this SINGLE connected component spans from top to bottom, it's a solid wall
            if reaches_top and reaches_bottom:
                return "solid", component

        # --- Pass 2: Perforated Wall Check ---
        # If no single connected shape spans top-to-bottom, evaluate vertical gaps row-by-row
        perforated_coords = set()
        row_has_solid = []

        for r in range(ceiling_row, floor_row + 1):
            has_solid = False
            for c in col_indices:
                if scene[r][c] in wall_set:
                    has_solid = True
                    perforated_coords.add((r, c))
            row_has_solid.append(has_solid)

        # Calculate max contiguous vertical gap (rows without any solid tile)
        max_gap = 0
        current_gap = 0
        for has_solid in row_has_solid:
            if not has_solid:
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0

        # Gaps <= 1 tile qualify as a perforated wall; gaps >= 2 tiles allow passage (no wall)
        if max_gap <= MAX_IMPASSABLE_GAP:
            return "perforated", perforated_coords

        return "none", set()

    left_cols = [0, 1, 2]
    right_cols = [width - 1, width - 2, width - 3]

    left_type, left_coords = evaluate_side(left_cols)
    right_type, right_coords = evaluate_side(right_cols)

    return left_type, left_coords, right_type, right_coords

def count_by_char(scene, char_to_id, chars, names, describe_absence, exclude=set()):
    """Counting phrase for one tile family, e.g. " two teleporters.", with its coordinates."""
    ids = [char_to_id[ch] for ch in chars if ch in char_to_id]
    if not ids:
        return "", []
    coords = [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row)
              if t in ids and (r, c) not in exclude]
    singular, plural = names
    phrase = count_caption_phrase(scene, ids, singular, plural,
                                  describe_absence=describe_absence, exclude=exclude)
    return phrase, coords


def describe_named_counts(scene, char_to_id, family_ids, names_table, family_names, describe_absence):
    """One counting phrase per named member of a tile family (enemies, power-ups, hazards),
    with its coordinates: " two mets. one sniper joe."

    Characters sharing a name in `names_table` are counted together (the two octopus battery
    orientations are one enemy). A family member the table does not cover is still counted
    under the family's own name, so a caption never silently drops a tile.
    """
    by_name = {}
    for ch, names in names_table.items():
        if ch in char_to_id:
            by_name[names] = by_name.get(names, "") + ch
    result = []
    named_ids = set()
    for names, chars in sorted(by_name.items()):
        phrase, coords = count_by_char(scene, char_to_id, chars, names, describe_absence=False)
        if phrase:
            result.append((phrase, coords))
        named_ids.update(char_to_id[ch] for ch in chars)

    rest = [i for i in family_ids if i not in named_ids]
    if rest:
        singular, plural = family_names
        phrase = count_caption_phrase(scene, rest, singular, plural, describe_absence=False)
        if phrase:
            result.append((phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in rest]))
    if describe_absence and not result:
        result.append((f" no {family_names[1]}.", []))
    return result


def strip_length_word(length, full_length):
    """"short", "long" or "full" for a hazard strip of `length` along a scene side of
    `full_length` tiles."""
    if length >= full_length:
        return "full"
    return "long" if length > SHORT_STRIP_MAX else "short"


def grouped_count_phrases(groups, names):
    """One counting phrase per key of `groups`, a dict of (label prefix, location phrase) ->
    list of cell sets: " two long lava pools at left." Returns (phrase, coordinates) tuples."""
    singular, plural = names
    result = []
    for (prefix, where), items in groups.items():
        count = len(items)
        label = plural if pluralize and count > 1 else singular
        if prefix:
            label = f"{prefix} {label}"
        text = f" {describe_quantity(count) if coarse_counts else count} {label}"
        text += f" {where}." if where else "."
        result.append((text, sorted(set().union(*items))))
    return result


def width_groups(items, scene_width, describe_locations):
    """Group lava pools or spike pits by width word and location, for grouped_count_phrases.
    One running the full width of the scene takes no location, since "full" already says it
    is everywhere."""
    groups = {}
    for cells in items:
        cols = [c for _, c in cells]
        item_w = max(cols) - min(cols) + 1
        where = None
        if describe_locations and coarse_locations and item_w < scene_width:
            where = describe_span(min(cols), max(cols))
        groups.setdefault((strip_length_word(item_w, scene_width), where), []).append(cells)
    return groups


def describe_lava_pools(scene, id_to_char, char_to_id, describe_locations, describe_absence):
    """Phrases for the lava pools, one per width and location: " one long lava pool at left."
    Returns (phrase, coordinates) tuples.
    """
    if LAVA_CHAR not in char_to_id:
        return []
    pools = find_objects(scene, id_to_char, LAVA_CHAR, set())
    result = grouped_count_phrases(width_groups(pools, len(scene[0]), describe_locations), LAVA_NAMES)
    if not result and describe_absence:
        result.append((f" no {LAVA_NAMES[1]}.", []))
    return result


def describe_spikes(scene, id_to_char, char_to_id, describe_locations, describe_absence):
    """Phrases for the spikes, by what they are part of (find_spike_groups): " one long spike pit
    at center. a few floor spikes. two wall spikes." Pits are described like lava pools; the
    other spikes are counted like enemies, by tile and without a location.
    Returns (phrase, coordinates) tuples.
    """
    if SPIKE_CHAR not in char_to_id:
        return []
    by_kind = {kind: [] for kind in SPIKE_NAMES}
    for kind, cells in find_spike_groups(scene, id_to_char, set()):
        by_kind[kind].append(cells)
    result = []
    for kind, names in SPIKE_NAMES.items():
        if kind == "pit":
            groups = width_groups(by_kind[kind], len(scene[0]), describe_locations)
        else:
            groups = {(None, None): by_kind[kind]} if by_kind[kind] else {}
        result += grouped_count_phrases(groups, names)
    if not result and describe_absence:
        result.append((" no spikes.", []))
    return result


def describe_tile_type_counts(scene, id_to_char, char_to_id, describe_absence,
                              describe_locations=False, already_accounted=frozenset()):
    """Phrases for the MMLV tile types that have no structural description of their own,
    each named on its own ("one boss door. one key door."). Multi-tile types are counted per
    placed object by find_objects rather than per tile.
    """
    result = []
    for chars, names, multi_tile in MMLV_TILE_TYPES:
        chars = [ch for ch in chars if ch in char_to_id]
        if not chars:
            continue
        if multi_tile:
            structures = find_objects(scene, id_to_char, chars, already_accounted)
            phrases = describe_structure_counts(structures, names, describe_locations=describe_locations)
            if not phrases and describe_absence:
                phrases = [(" no " + names[1] + ".", set())]
            result.extend((p, sorted(c)) for p, c in phrases)
        else:
            result.append(count_by_char(scene, char_to_id, chars, names, describe_absence,
                                        exclude=already_accounted))
    return result


def assign_caption(scene, id_to_char, char_to_id, tile_descriptors, describe_locations, describe_absence,
                   data=None, debug=False, return_details=False):
    """Assigns a caption to a level scene based on its contents.

    Every enemy, power-up, hazard and gimmick is named by its type ("two mets. one large life
    energy. several spikes. one boss door.") rather than pooled into its family.
    """
    already_accounted = set()
    details = {} if return_details else None
    ladder_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'climbable' in value]
    enemy_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'enemy' in value]
    powerup_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'powerup' in value]
    empty_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'empty' in value] #Used for water ratio calculation
    water_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'water' in value] #Used for water ratio calculation
    hazard_ids = [char_to_id[key] for key, value in tile_descriptors.items() if 'hazard' in value]
    wall_ids = [char_to_id[key] for key, value in tile_descriptors.items() if (('solid' in value) and ('penetrable' not in value) and ("hazard" not in value))]
    disappearing_ids = [char_to_id["A"]] if "A" in char_to_id else [] #There's nothing unique about the descriptors for disappearing blocks, so we just set it here
    #Ideas:
    #Walls for each size/exit directions
    #Some kind of data transfer telling us which way the level is moving
        #DONE Encode "enter:", "exit:", and "blocked:", all giving us a direction
    #Check for ladders, enemies, powerups, water/air, spikes, moving/disappearing blocks
        #DONE Ladders: count number of vertical strips
        #DONE enemies: same as mario, raw count
        #DONE powerups: same 
        #DONE water:a little, a lot, half, mostly, all: mesures water/air ratio, 0-10% water, 10-40%, 40-60%, 60-99%, 100% respectivly
        #DONE Spikes: a few:0-5, a lot:6+
        #DONE Moving platforms: one, two, several, for 1, 2, 3+ continuous horizantal platforms
        #DONE Disappearing blocks: a few: 0-3, a lot:4+
    #Base checks, mostly unchanged
        #DONE Platforms (slightly expand definition of a platform)
        #DONE Loose blocks (same as mario)    
    

    def add_to_caption(phrase, contributing_blocks):
        nonlocal caption
        #if phrase and "ceiling" in phrase:
        #    raise ValueError(f"{phrase} {contributing_blocks}")

        if phrase:
            caption += phrase
            if return_details and details is not None and contributing_blocks != None:
                details[phrase.strip()] = contributing_blocks

    caption = ""

    
    if data != None: #Some systems cant give us this extra data
        #Add captions from encoded data
        entrance_direction = data['entrance_direction']
        exit_direction = data['exit_direction']
        
        add_to_caption(f" entrance direction {entrance_direction.lower()}.", None)
        add_to_caption(f" exit direction {exit_direction.lower()}.", None)
    else:
        entrance_direction = None
        exit_direction = None


    # Count enemies, power-ups and hazards, each type under its own name. Spikes and lava are
    # described by shape below, so they are left out of the hazard counts, and "no hazards"
    # only applies when there are none of those either.
    strip_ids = {char_to_id[ch] for ch in (SPIKE_CHAR, LAVA_CHAR) if ch in char_to_id}
    has_strips = any(t in strip_ids for row in scene for t in row)
    for family_ids, names_table, family_names, absence in (
            (enemy_ids, ENEMY_NAMES, ("enemy", "enemies"), describe_absence),
            (powerup_ids, POWERUP_NAMES, ("powerup", "powerups"), describe_absence),
            ([i for i in hazard_ids if i not in strip_ids], HAZARD_NAMES, ("hazard", "hazards"),
             describe_absence and not has_strips)):
        for phrase, coords in describe_named_counts(scene, char_to_id, family_ids, names_table,
                                                    family_names, absence):
            add_to_caption(phrase, coords)

    for describe in (describe_spikes, describe_lava_pools):
        for phrase, coords in describe(scene, id_to_char, char_to_id, describe_locations, describe_absence):
            add_to_caption(phrase, coords)


    # Count disappearing blocks
    disappearing_phrase = count_caption_phrase(scene, disappearing_ids, "disappearing block", "disappearing blocks", describe_absence=describe_absence)
    add_to_caption(disappearing_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in disappearing_ids])

    # Count the MMLV tile types that the Mega Man captioner has no phrase for at all
    for phrase, coords in describe_tile_type_counts(
            scene, id_to_char, char_to_id, describe_absence,
            describe_locations=describe_locations):
        add_to_caption(phrase, coords)

    # Key doors are tagged solid, so without this the platform and block passes below would
    # describe a door a second time, as a platform or a block cluster.
    door_chars = {ch for ch, value in tile_descriptors.items() if "door" in value}
    already_accounted.update((r, c) for r, row in enumerate(scene) for c, t in enumerate(row)
                             if id_to_char[t] in door_chars)

    #Count water
    water_phrase = find_water_caption(scene, empty_ids, water_ids, describe_absence)
    add_to_caption(water_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in water_ids])


    # Ceiling
    ceiling_row = None
    if (exit_direction is not None and exit_direction.lower() in ("left", "right")): # Only track ceiling if we're moving horizantally (exit_direction is stored as uppercase enum name, e.g. "RIGHT")
        ceiling_row = 2 #Define this here so we don't ignore platforms on row 2 later if we're moving vertically
        ceiling_tiles = [(ceiling_row, c) for c, t in enumerate(scene[ceiling_row]) if t in wall_ids]
        ceiling_phrase = analyze_ceiling(scene, wall_ids, describe_absence, ceiling_row=ceiling_row)
        already_accounted.update(ceiling_tiles)
        add_to_caption(ceiling_phrase, ceiling_tiles)

    # Floor. A ladder tile at the floor is not a fall-through gap, so it counts as floor.
    non_gap_ids = set(wall_ids) | set(ladder_ids)
    floor = find_floor(scene, non_gap_ids)
    if floor is not None:
        floor_row = floor.row
        floor_rows = floor.rows
        floor_phrase = describe_floor_occupancy(floor.occupancy, describe_absence)
        already_accounted.update(floor.accounted)
        floor_tiles = sorted(floor.surface)
    else:
        # Nothing near the bottom is standable, so describe the last row as before.
        floor_row = len(scene) - 1
        floor_rows = [floor_row]
        floor_phrase = analyze_floor(
            scene,
            wall_ids,
            describe_absence=describe_absence,
            floor_row=floor_row,
            ladder_ids=ladder_ids
        )
        floor_tiles = [(floor_row, c) for c, t in enumerate(scene[floor_row]) if t in wall_ids]
        already_accounted.update(floor_tiles)
    add_to_caption(floor_phrase, floor_tiles)

    # --- Edge Wall Detection ---
    # Assumes the ceiling is on row 2, which works for 16x16 scenes but not for larger ones.
    # The playable area ends at the floor, so a wall only has to span down to it.
    left_type, left_wall_coords, right_type, right_wall_coords = detect_edge_walls(
        scene,
        wall_ids,
        ceiling_row=ceiling_row if ceiling_row else 2, # TODO: Generalize
        floor_row=floor_row
    )

    # Format Left Wall Caption
    if left_type == "solid":
        add_to_caption(" left wall.", list(left_wall_coords))
    elif left_type == "perforated":
        add_to_caption(" perforated left wall.", list(left_wall_coords))
    elif describe_absence:
        add_to_caption(" no left wall.", [])

    # Format Right Wall Caption
    if right_type == "solid":
        add_to_caption(" right wall.", list(right_wall_coords))
    elif right_type == "perforated":
        add_to_caption(" perforated right wall.", list(right_wall_coords))
    elif describe_absence:
        add_to_caption(" no right wall.", [])

    # Platforms
    # Count moving platforms
    moving_plat_lines = find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor="moving", min_run_length=1, require_above_below_not_solid=True, already_accounted=already_accounted, exclude_rows=[ceiling_row] + floor_rows)
    # Group the runs by the tile they are built from. A run is named after its leftmost
    # tile; runs mixing two kinds of moving block are too rare to be worth splitting.
    by_kind = {}
    for y, start_x, end_x in moving_plat_lines:
        kind = id_to_char[scene[y][start_x]]
        by_kind.setdefault(kind, []).append((y, start_x, end_x))
    for kind, lines in sorted(by_kind.items()):
        singular, _plural = MOVING_PLATFORM_NAMES.get(kind, _pair("moving platform"))
        # describe_horizontal_lines pluralises by appending an s, which is correct for
        # every name in MOVING_PLATFORM_NAMES.
        phrase = describe_horizontal_lines(lines, singular, describe_locations, describe_absence=False)
        add_to_caption(phrase, [(y, x) for y, s, e in lines for x in range(s, e + 1)])
    if not moving_plat_lines and describe_absence:
        add_to_caption(" no moving platforms.", [])

    #Count regular platforms
    platform_lines = find_horizontal_lines(scene, id_to_char, tile_descriptors, target_descriptor="solid", min_run_length=2, require_above_below_not_solid=True, already_accounted=already_accounted, exclude_rows=[ceiling_row] + floor_rows)
    #print("after platform_lines", (10,0) in already_accounted)
    platform_phrase = describe_horizontal_lines(platform_lines, "platform", describe_locations, describe_absence=describe_absence)
    add_to_caption(platform_phrase, [(y, x) for y, start_x, end_x in platform_lines for x in range(start_x, end_x + 1)])


    # Solid structures
    
    #Count ladders
    ladder_phrases = find_ladders(scene, ladder_ids, already_accounted, describe_absence, floor_row=floor_row)
    for phrase, coords in ladder_phrases:
        add_to_caption(phrase, coords)

    # Breakable blocks form their own structures. This runs before find_solid_structures so a
    # 2x2 of breakable blocks reads as one breakable block cluster instead of vanishing into
    # the terrain it is embedded in.
    breakable_chars = [ch for ch, desc in tile_descriptors.items()
                       if "breakable" in desc and "door" not in desc]
    breakable_structures = find_char_structures(scene, id_to_char, breakable_chars, already_accounted)
    for phrase, coords in describe_tile_structures(
            breakable_structures, "breakable block", "breakable blocks",
            describe_locations=describe_locations,
            describe_absence=describe_absence):
        add_to_caption(phrase, sorted(coords))
        already_accounted.update(coords)

    structures = find_solid_structures(scene, id_to_char, tile_descriptors, already_accounted)
    # Pass the rows we actually found: describe_structures decides what is a tower by
    # whether a structure touches the floor, and defaulting to the last row misclassifies
    # every scene whose floor is somewhere else.
    structure_phrase = describe_structures(
        structures,
        ceiling_row=ceiling_row if ceiling_row is not None else CEILING,
        floor_row=floor_row,
        describe_locations=describe_locations,
        describe_absence=describe_absence,
        debug=debug
    )

    for phrase, coords in structure_phrase:
        add_to_caption(phrase, coords)

    #print(already_accounted)
    loose_block_phrase = count_caption_phrase(scene, wall_ids, "loose block", "loose blocks", describe_absence=describe_absence, exclude=already_accounted)
    add_to_caption(loose_block_phrase, [(r, c) for r, row in enumerate(scene) for c, t in enumerate(row) if t in wall_ids and (r, c) not in already_accounted])

    return (caption.strip(), details) if return_details else caption.strip()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate captions for Mega Man Maker (MMLV) scenes")
    parser.add_argument("--dataset", required=True, help="json with level scenes")
    
    parser.add_argument("--tileset", default=common_settings.MMLV_TILESET, help="Descriptions of individual tile types")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--no_describe_locations", dest="describe_locations",
                        action="store_false", default=True,
                        help="Leave the coarse position (\"at left\") off the phrases that carry one, "
                             "such as platforms, block clusters and breakable blocks. Positions are "
                             "included by default.")
    parser.add_argument("--caption-mode", choices=["legacy", "keyed"], default="legacy",
                        help="Output schema. 'legacy' (default) writes the single 'caption' field. 'keyed' writes the caption as a "
                             "one-element list under --caption-key, so a scene can carry captions from several sources at once. Both "
                             "modes copy all other input attributes (metadata and captions from other sources) to the output.")
    parser.add_argument("--caption-key", default="deterministic_captions",
                        help="Key to store the caption list under when --caption-mode keyed. Default: deterministic_captions")
    global args
    args = parser.parse_args()

    dataset_file = args.dataset
    tileset_file = args.tileset
    output_file = args.output

    if not os.path.isfile(dataset_file) or not os.path.isfile(tileset_file):
        print("Error: One or more input files do not exist.")
        sys.exit(1)

    generate_captions(dataset_file, tileset_file, output_file, args.describe_locations, args.describe_absence,
                      caption_mode=args.caption_mode, caption_key=args.caption_key)
