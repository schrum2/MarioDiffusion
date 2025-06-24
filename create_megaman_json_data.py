import json
import argparse
from pathlib import Path
import util.common_settings as common_settings
from captions.util import extract_tileset
from create_level_json_data import load_levels, load_tileset


def parse_args():
    parser = argparse.ArgumentParser(description="Create level json files for megaman")
    
    parser.add_argument('--tileset', default='..\\TheVGLC\\MegaMan\\MM.json', help='Path to the tile set JSON')
    parser.add_argument('--levels', default='..\\TheVGLC\\MegaMan\\Enhanced', help='Directory containing level text files')
    #parser.add_argument('--output', required=True, help='Path to the output JSON file')

    parser.add_argument('--target_height', type=int, default=common_settings.MEGAMAN_HEIGHT, help='Target output height (e.g., 16 or 14)')
    parser.add_argument('--target_width', type=int, default=common_settings.MEGAMAN_WIDTH, help='Target output width (e.g., 16)')


    return parser.parse_args()


def main():

    args = parse_args()

    levels = load_levels(args.levels)
    tileset = load_tileset(args.tileset, "-")
    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(args.tileset)
    for level in levels:
        parse_level(level, args.target_width, args.target_height)


#Parses through one complete level
def parse_level(level, width, height):
    startx, starty = find_start(level)
    sample=get_sample_from_idx(level, startx, starty, width, height)
    for row in sample:
        print(row)
    print("\n\n")


#Finds the spawn sample to begin searching
def find_start(level):
    for i in range(len(level)):
        if level[i][0]!='@':
            return 0, i
    return 0, 0


#Gets a full level sample of the desired size from the top left corner
def get_sample_from_idx(level, col, row, width, height):
    
    #Make sure the level sample is in bounds
    if row<0 or col<0 or width<0 or height<0:
        raise ValueError(f"Row ({row}), collumn ({col}), width ({width}), and height ({height}) all must be positive.")
    if (row + height)>len(level) or (col+width)>len(level[0]):
        raise ValueError(f"This level sample is out of bounds at the bottom or right, with height index {row+height}/{len(level)} and width index {col+width}/{len(level[0])}.")
    
    sample = []
    for row in level[row:row+height]:
        sample.append(row[col:col+height])
    
    return sample



if __name__ == "__main__":
    main()