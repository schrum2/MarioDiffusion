import json
import argparse
from pathlib import Path
import util.common_settings as common_settings
from captions.util import extract_tileset

def load_tileset(tileset_path):
    """
    Loads a tileset from a JSON file and maps tile characters to unique IDs.

    Args:
        tileset_path (str): Path to the JSON file containing the tileset data.

    Returns:
        dict: A dictionary mapping tile characters to unique integer IDs.
    """
    # Use the project's shared tileset loader so the character->id assignment matches
    # every other dataset. extract_tileset deliberately preserves the tileset file's
    # order for MegaMan tilesets (so '@'=null is id 1, etc.); sorting the characters
    # here instead produced a different, incompatible id scheme that decoded to the
    # wrong tiles.
    _, _, tile_to_id, _ = extract_tileset(tileset_path)
    global extra_tile
    if extra_tile not in tile_to_id:
        raise ValueError(
            f"Tileset {tileset_path} has no '{extra_tile}' tile, so characters outside the "
            f"tileset cannot be encoded. Add it to the tileset or choose a different extra_tile."
        )
    return tile_to_id

def load_char_map(char_map_path, tile_to_id):
    """
    Loads a character remapping from a JSON file.

    The map translates characters found in the raw VGLC level data into the
    characters of the (simplified) tileset, e.g. mapping every enemy character
    in the full MegaMan tileset to the single enemy character "a" in the
    simplified tileset. Without this, characters absent from the simplified
    tileset are silently treated as the empty/extra tile, which discards
    enemies, powerups, etc.

    Args:
        char_map_path (str): Path to a JSON file of {source_char: target_char}.
        tile_to_id (dict): The id mapping for the (simplified) tileset, used to
            validate that every target character is actually in the tileset.

    Returns:
        dict: A dictionary mapping source characters to target characters. Empty
            if char_map_path is None.
    """
    if char_map_path is None:
        return {}

    with open(char_map_path, 'r') as f:
        char_map = json.load(f)

    # Fail loudly on a target that the tileset can't represent, so a typo in the
    # map doesn't silently fall through to the extra tile.
    bad_targets = {src: tgt for src, tgt in char_map.items() if tgt not in tile_to_id}
    if bad_targets:
        raise ValueError(
            f"Character map targets not present in the tileset: {bad_targets}. "
            f"Valid target characters are: {sorted(tile_to_id.keys())}"
        )

    return char_map


def load_levels(levels_dir):
    """
    Loads levels from text files in a specified directory.

    Args:
        levels_dir (str): Path to the directory containing level text files.

    Returns:
        list: A list of levels, where each level is represented as a list of strings.
    """
    levels = set()
    for file in sorted(Path(levels_dir).glob("*.txt")):
        with open(file, 'r') as f:
            level = tuple(line.strip() for line in f if line.strip())
            # if level:  # Only add non-empty levels
            #     print(f"Loaded level from {file.name} with {len(level)} rows")
            #     for row in level:
            #         print(f"  {row}")
            levels.add(level)
    return [list(level) for level in levels]  # Convert back to list-of-lists for compatibility


def load_scene_dataset(dataset_path):
    """
    Loads already-encoded integer scenes from one of this repo's scene datasets,
    e.g. datasets/MM_LevelsAndCaptions-regular-train.json. Each entry has a
    "scene" that is a 2D grid of integer tile ids, so no tileset/character lookup
    is needed and the ids already match the diffusion model's tile space.

    Args:
        dataset_path (str): Path to the scene dataset JSON file.

    Returns:
        list: A list of scenes, each a 2D list of integer tile ids.
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return [entry["scene"] if isinstance(entry, dict) else entry for entry in data]


def pad_and_sample(level, tile_to_id, window_size, char_map=None, unmapped_chars=None):
    """
    Extracts tile samples of a specified window size from a level.

    Args:
        level (list): A 2D list representing the level layout.
        tile_to_id (dict): A dictionary mapping tile characters to unique IDs.
        window_size (int): The size of the square window to extract.
        char_map (dict): Optional {source_char: target_char} remapping applied to
            each character before the id lookup (see load_char_map).
        unmapped_chars (set): Optional set that collects any characters that were
            neither in the tileset nor in char_map and so fell back to the extra
            tile. Used to warn the caller that the map is incomplete.

    Returns:
        list: A list of 2D lists, each representing a sampled window of tiles.
    """
    if char_map is None:
        char_map = {}
    height = len(level)
    width = len(level[0])
    samples = set()  # Use a set to avoid duplicates

    # Iterate through the level, extracting tiles that fit entirely within bounds
    for y in range(0, height - window_size + 1):
        for x in range(0, width - window_size + 1):
            sample = []
            for row_idx in range(y, y + window_size):
                window_row = []
                for col_idx in range(x, x + window_size):
                    raw_char = level[row_idx][col_idx]
                    # Remap the raw VGLC character to the tileset's character set
                    # (identity if it has no entry), then look up its id.
                    char = char_map.get(raw_char, raw_char)
                    if char not in tile_to_id and unmapped_chars is not None:
                        unmapped_chars.add(raw_char)
                    tile_id = tile_to_id.get(char, tile_to_id[extra_tile])
                    window_row.append(tile_id)
                sample.append(tuple(window_row))  # Convert row to tuple
            samples.add(tuple(sample))  # Convert sample to tuple of tuples before adding

    print(f"Extracted {len(samples)} unique samples.")
    return samples


def sample_scene(scene, window_size):
    """
    Same idea as pad_and_sample but for an already-encoded integer scene: the
    cells are tile ids, so there is no character/tileset lookup to do.

    Args:
        scene (list): A 2D list of integer tile ids.
        window_size (int): The size of the square window to extract.

    Returns:
        set: A set of windows, each a tuple of tuples of integer tile ids.
    """
    height = len(scene)
    width = len(scene[0])
    samples = set()

    for y in range(0, height - window_size + 1):
        for x in range(0, width - window_size + 1):
            sample = tuple(tuple(scene[row_idx][x:x + window_size])
                           for row_idx in range(y, y + window_size))
            samples.add(sample)

    print(f"Extracted {len(samples)} unique samples.")
    return samples

def main(tileset_path, levels_dir, output_path, window_size, char_map_path=None, dataset_path=None):
    """
    Orchestrates the process of loading levels, generating samples, and saving them to a JSON file.

    When dataset_path is given the windows are sliced straight out of an existing
    integer-encoded scene dataset (the path used for Mario Maker here); otherwise
    the original VGLC flow is used, reading ASCII level files and a tileset.

    Args:
        tileset_path (str): Path to the JSON file containing the tileset data.
        levels_dir (str): Path to the directory containing level text files.
        output_path (str): Path to save the output JSON file.
        window_size (int): The size of the square window to extract.
        char_map_path (str): Optional path to a {source_char: target_char} JSON
            file remapping raw VGLC characters onto the tileset (see load_char_map).
        dataset_path (str): Optional path to an integer-encoded scene dataset; when
            set, tileset_path/levels_dir/char_map_path are ignored.

    Returns:
        None
    """
    dataset = []
    unique_set = set()

    if dataset_path:
        scenes = load_scene_dataset(dataset_path)
        for scene in scenes:
            samples = sample_scene(scene, window_size)
            for sample in samples:
                dataset.append([list(row) for row in sample])
                unique_set.add(sample)
    else:
        tile_to_id = load_tileset(tileset_path)
        char_map = load_char_map(char_map_path, tile_to_id)
        levels = load_levels(levels_dir)

        unmapped_chars = set()
        for level in levels:
            samples = pad_and_sample(level, tile_to_id, window_size, char_map=char_map, unmapped_chars=unmapped_chars)
            for sample in samples:
                dataset.append([list(row) for row in sample])
                unique_set.add(sample)

        # These characters appeared in the level data but were neither in the tileset
        # nor in the character map, so they were encoded as the extra tile ('{extra_tile}').
        # That is usually a sign the character map is incomplete.
        if unmapped_chars:
            print(
                f"WARNING: {len(unmapped_chars)} character(s) not in the tileset or character map "
                f"were encoded as the extra tile '{extra_tile}': {sorted(unmapped_chars)}"
            )

    print(f"Total samples: {len(dataset)}")
    print(f"Unique samples: {len(unique_set)}")


    # Convert back to lists for JSON serialization
    dataset = [ [list(row) for row in sample] for sample in unique_set ]

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Reload the saved JSON file and print the length of the loaded list
    with open(output_path, 'r') as f:
        loaded_dataset = json.load(f)
    print(f"Length of loaded dataset: {len(loaded_dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tileset', default=common_settings.MARIO_TILESET, help='Path to the tile set JSON')
    parser.add_argument('--levels', default=r'..\TheVGLC\Super Mario Bros\Processed', help='Directory containing level text files')
    parser.add_argument('--output', required=True, help='Path to the output JSON file')
    parser.add_argument('--tile_size', type=int, required=False, help='Size of the tile (window) to extract')
    parser.add_argument('--char_map', default=None, help='Optional path to a {source_char: target_char} JSON that remaps raw VGLC characters onto the (simplified) tileset')
    parser.add_argument('--from_dataset', default=None, help='Path to an integer-encoded scene dataset (e.g. datasets/MM_LevelsAndCaptions-regular-train.json). When set, windows are sliced from the scenes and --tileset/--levels/--char_map are ignored.')
    args = parser.parse_args()

    global extra_tile
    extra_tile = "-"

    # Add debug prints
    print(f"Loading tileset from: {args.tileset}")
    print(f"Loading levels from: {args.levels}")
    print(f"Output will be saved to: {args.output}")
    print(f"Using tile size: {args.tile_size}")
    print(f"Using character map: {args.char_map}")

    # Call main with parsed arguments
    main(args.tileset, args.levels, args.output, args.tile_size, args.char_map, args.from_dataset)
