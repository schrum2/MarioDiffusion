import json
import argparse
from pathlib import Path
import util.common_settings as common_settings
from captions.util import extract_tileset
from create_level_json_data import load_levels, load_tileset
from enum import Enum


#Needed to identify the direction of the sample
class Direction(Enum):
    UP=1
    DOWN=2
    RIGHT=3
    LEFT=4



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
    startx, starty = find_start(level, width, height)
    sample=get_sample_from_idx(level, startx, starty, width, height)
    sample2, new_direction = move_scene(level, startx, starty, width, height, Direction.UP)
    for row in sample:
        print(row)
    for row in sample2:
        print(row)
    print("\n\n")


#Finds the spawn sample to begin searching
def find_start(level, width, height):
    start_y=-1
    start_x=-1

    #Loop through every row to find the spawn location
    for i in range(len(level)):
        if level[i].find('P')!=-1:
            start_y=i
            start_x=level[i].find('P')
            break
    
    if start_y==-1:
        raise ValueError("Spawn location not found!")
    

    #Continue searching down for the bottom of the level or more null chars
    #We do this to get the full level scene, not just the spawn point and up
    lowest_possible_start = min(len(level), start_y+height)
    lowest_found = False
    for i in range(start_y, lowest_possible_start):
        if level[i][start_x]=='@':
            start_y=i
            start_y=start_y-height #This is needed because we expect a top left index, not a bottom left
            lowest_found=True
            break
    

    #Check to see if we didn't find a lower null char (Meaning we hit the bottom of the level, or the level keeps going down awhile)
    if not lowest_found:
        #Did we reach the bottom of the level?
        if lowest_possible_start==len(level):
            start_y=lowest_possible_start
            start_y=start_y-height
        #If not, the level is vertical downwards, so we need to go up to reach the top
        else:
            #Pretty much the same sequence of checks again, just going up this time, this should only rarely be needed
            highest_possible_start=max(start_y-height, 0)
            heighest_found=False
            for i in range(start_y, highest_possible_start, -1):
                if level[i][start_x]=='@':
                    start_y=i
                    break
            
            if not heighest_found:
                start_y=highest_possible_start

    

    #Start at the left edge if close enough
    if start_x<width:
        start_x=0
    else:
        start_x=start_x-width
    
    return start_x, start_y


#Move the sliding window one block
def move_scene(level, old_x_idx, old_y_idx, width, height, direction: Direction):
    mid_x = (old_x_idx + (old_x_idx+width))//2 #Gets the central x-coordinate of the sample
    mid_y = (old_y_idx + (old_y_idx+height))//2 #Gets the central y-coordinate of the sample


    #Changedir if: 
        #The right wall of the prev. sample is only wall
        #The spot we would be moving into has null tokens
    #If the right is blocked (wall)
    #Move the scene one block to the right
    if direction == Direction.UP:
        print("UP")

        y_idx = old_y_idx - 1
        x_idx = old_x_idx

    if direction == Direction.DOWN:
        print("DOWN")

        y_idx = old_y_idx + 1
        x_idx = old_x_idx

    if direction == Direction.RIGHT:
        print("RIGHT")

        x_idx = old_x_idx + 1
        y_idx = old_y_idx

    if direction == Direction.LEFT:
        print("LEFT")

        x_idx = old_x_idx - 1
        y_idx = old_y_idx

    """#If we've moved past the right edge, move down a row
    if x_idx + width > len(level[0]):
        x_idx = 0
        y_idx += 1

    #If we've moved past the bottom edge, stop
    if y_idx + height > len(level):
        return None"""

    return get_sample_from_idx(level, x_idx, y_idx, width, height), direction


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