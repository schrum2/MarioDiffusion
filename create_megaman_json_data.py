import json
import argparse
from pathlib import Path
import util.common_settings as common_settings
from captions.util import extract_tileset
from create_level_json_data import load_levels, load_tileset
from enum import Enum


#Needed to identify the direction of the sample
class Direction(Enum):
    UP=0
    RIGHT=1
    DOWN=2
    LEFT=3



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

    sample_arr=[
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16]
    ]

    #print(sample_arr[1][1:3])
    #print([x[1] for x in sample_arr[1:3]])

    levels = load_levels(args.levels)
    tileset = load_tileset(args.tileset, "-")
    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(args.tileset)
    null_chars = [key for key, value in tile_descriptors.items() if 'null' in value]
    wall_chars = [key for key, value in tile_descriptors.items() if (('solid' in value) and ('penetrable' not in value))]
    print(null_chars)
    print(wall_chars)

    parse_level(levels[2], args.target_width, args.target_height, null_chars, wall_chars)

    #for level in levels:
    #    parse_level(level, args.target_width, args.target_height, null_chars, wall_chars)


#Parses through one complete level
def parse_level(level, width, height, null_chars=['@'], wall_chars=['#']):
    x_idx, y_idx = find_start(level, width, height)
    sample=get_sample_from_idx(level, x_idx, y_idx, width, height)
    for row in sample:
        print(row)
    print("")
    new_direction = Direction.RIGHT
    for i in range(70):
        print("")
        x_idx, y_idx, new_direction = move_scene(level, x_idx, y_idx, width, height, new_direction, null_chars, wall_chars)
        print("Level: ")
        sample2 = get_sample_from_idx(level, x_idx, y_idx, width, height)
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
def move_scene(level, old_x_idx, old_y_idx, width, height, direction: Direction, null_chars=['@'], wall_chars=['#']):


    #Changedir if: 
        #The right wall of the prev. sample is only wall
        #The spot we would be moving into has null tokens
    #If the right is blocked (wall)
    #Move the scene one block to the right
    if direction == Direction.UP and is_up_possible(level, old_x_idx, old_y_idx, width, height, null_chars, wall_chars):
        y_idx = old_y_idx - 1
        x_idx = old_x_idx
        next_direction=direction

    elif direction == Direction.DOWN and is_down_possible(level, old_x_idx, old_y_idx, width, height, null_chars, wall_chars):
        y_idx = old_y_idx + 1
        x_idx = old_x_idx
        next_direction=direction

    elif direction == Direction.RIGHT and is_right_possible(level, old_x_idx, old_y_idx, width, height, null_chars, wall_chars):
        x_idx = old_x_idx + 1
        y_idx = old_y_idx
        next_direction=direction

    elif direction == Direction.LEFT and is_left_possible(level, old_x_idx, old_y_idx, width, height, null_chars, wall_chars):
        x_idx = old_x_idx - 1
        y_idx = old_y_idx
        next_direction=direction

    else:
        x_idx, y_idx, next_direction = change_direction(level, old_x_idx, old_y_idx, width, height, direction, null_chars, wall_chars)

    print(next_direction.name)

    return x_idx, y_idx, next_direction


# Changes the current direction to be vertical and moves the index one block in that direction
def change_direction(level, old_x_idx, old_y_idx, width, height, direction: Direction, null_chars, wall_chars):
    #Method calls to automate
    def move_down():
        new_direction=Direction.DOWN
        y_idx=old_y_idx+1
        return old_x_idx, y_idx, new_direction
    
    def move_up():
        new_direction=Direction.UP
        y_idx=old_y_idx-1
        return old_x_idx, y_idx, new_direction
    
    def move_right():
        new_direction=Direction.RIGHT
        x_idx=old_x_idx+1
        return x_idx, old_y_idx, new_direction
    
    def move_left():
        new_direction=Direction.LEFT
        x_idx=old_x_idx-1
        return x_idx, old_y_idx, new_direction
    


    if direction.value%2==1:
        up_possible = is_up_possible(level, old_x_idx, old_y_idx, width, height, null_chars, wall_chars)
        down_possible = is_down_possible(level, old_x_idx, old_y_idx, width, height, null_chars, wall_chars)
        
        if up_possible and down_possible:
            raise ValueError("I don't know which way to go!")
        elif not up_possible and not down_possible:
            raise ValueError("Both directions are impassible!")
        elif up_possible:
            return move_up()
        else:
            return move_down()
    else:
        left_possible = is_left_possible(level, old_x_idx, old_y_idx, width, height, null_chars, wall_chars)
        right_possible = is_right_possible(level, old_x_idx, old_y_idx, width, height, null_chars, wall_chars)
        
        if left_possible and right_possible:
            raise ValueError("I don't know which way to go!")
        elif not left_possible and not right_possible:
            raise ValueError("Both directions are impassible!")
        elif left_possible:
            return move_left()
        else:
            return move_right()


def is_up_possible(level, old_x_idx, old_y_idx, width, height, null_chars, wall_chars):
    #If we're at the top of the screen
    if old_y_idx <= 0:
        return False
    
    top_row = level[old_y_idx][old_x_idx:old_x_idx+width]
    above_top_row = level[old_y_idx-1][old_x_idx:old_x_idx+width]

    if any(x in above_top_row for x in null_chars):
        return False
    
    if not any(x not in wall_chars for x in top_row):
        return False
    
    return True

def is_down_possible(level, old_x_idx, old_y_idx, width, height, null_chars, wall_chars):
    #If we're at the top of the screen
    if old_y_idx+height >= len(level):
        return False
    
    bottom_row = level[old_y_idx+height-1][old_x_idx:old_x_idx+width]
    below_bottom_row = level[old_y_idx+height][old_x_idx:old_x_idx+width]

    if any(x in below_bottom_row for x in null_chars):
        return False
    
    if not any(x not in wall_chars for x in bottom_row):
        return False
    
    return True

def is_left_possible(level, old_x_idx, old_y_idx, width, height, null_chars, wall_chars):
    #If we're at the top of the screen
    if old_x_idx <= 0:
        return False
    
    left_col = [x[old_x_idx] for x in level[old_y_idx:old_y_idx+height]]
    left_of_left_col = [x[old_x_idx-1] for x in level[old_y_idx:old_y_idx+height]]

    if any(x in left_of_left_col for x in null_chars):
        return False
    
    if not any(x not in wall_chars for x in left_col):
        return False
    
    return True

def is_right_possible(level, old_x_idx, old_y_idx, width, height, null_chars, wall_chars):
    #If we're at the top of the screen
    if old_x_idx+height >= len(level[0]):
        return False
    
    right_col = [x[old_x_idx+width-1] for x in level[old_y_idx:old_y_idx+height]]
    right_of_right_col = [x[old_x_idx+width] for x in level[old_y_idx:old_y_idx+height]]
    print("\n\n")
    print(right_col)
    print(right_of_right_col)
    print("\n\n")
    if any(x in right_of_right_col for x in null_chars):
        print("A")
        return False
    
    if not any(x not in wall_chars for x in right_col):
        return False
    
    return True

#Gets a full level sample of the desired size from the top left corner
def get_sample_from_idx(level, col, row, width, height):
    
    #Make sure the level sample is in bounds
    if row<0 or col<0 or width<0 or height<0:
        raise ValueError(f"Row ({row}), collumn ({col}), width ({width}), and height ({height}) all must be positive.")
    if (row + height)>len(level) or (col+width)>len(level[0]):
        raise ValueError(f"This level sample is out of bounds at the bottom or right, with height index {row+height}/{len(level)} and width index {col+width}/{len(level[0])}.")
    
    sample = []
    for row in level[row:row+height]:
        sample.append(row[col:col+width])
    
    return sample



if __name__ == "__main__":
    main()