import json
import argparse
from pathlib import Path
import util.common_settings as common_settings
from captions.util import extract_tileset
from create_level_json_data import load_levels, load_tileset
from enum import Enum


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
    def is_possible_to_move_direction(self, level, check_for_walls = False):
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
            walls_index = index-self.offset_for_axis #We only want the wall at the end of the row, not the row behind it
            walls_row = self.get_row_or_col(level, walls_index)
            if not any(x not in level.wall_chars for x in walls_row):
                return False
        
        return True #Base case, fires if we're not out of bounds, there's no null ahead, and optionally there's no wall blocking us



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

    #We literally only need level overrides for 1-7, every other level parses as expected
    overrides_1_7 = [120, 121, 122, 123, 182] #Needed to avoid an early turn leading to a split path, and to prevent the level from turning back around to go back to the start


    for i in range(len(levels)):
        if i==7: #We need to do some slight overrides on 1-7 to make the level functional
            parse_level(levels[i], args.target_width, args.target_height, null_chars, wall_chars, print_at_corners=True, change_direction_overrides=overrides_1_7)
        else:
            parse_level(levels[i], args.target_width, args.target_height, null_chars, wall_chars, print_at_corners=True)
        print(i)

    #for level in levels:
    #    parse_level(level, args.target_width, args.target_height, null_chars, wall_chars)


#Parses through one complete level
def parse_level(level, width, height, null_chars=['@'], wall_chars=['#'], start_direction=Direction.RIGHT, print_at_corners=False, change_direction_overrides=[]):
    level_sample=LevelSample(level, width, height, null_chars, wall_chars, start_direction=start_direction, print_at_corners=print_at_corners, change_direction_overrides=change_direction_overrides)

    moving=True
    while moving:
        moving=level_sample.move_step()
    
    level_sample.print_sample()



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
        raise ValueError("Spawn location not found!")
    

    #Continue searching down for the bottom of the level or more null chars
    #We do this to get the full level scene, not just the spawn point and up
    lowest_possible_start = min(len(level_sample.level), start_y+level_sample.height)
    lowest_found = False
    for i in range(start_y, lowest_possible_start):
        if level_sample.level[i][start_x]=='@':
            start_y=i
            start_y=start_y-level_sample.height #This is needed because we expect a top left index, not a bottom left
            lowest_found=True
            break
    

    #Check to see if we didn't find a lower null char (Meaning we hit the bottom of the level, or the level keeps going down awhile)
    if not lowest_found:
        #Did we reach the bottom of the level?
        if lowest_possible_start==len(level_sample.level):
            start_y=lowest_possible_start
            start_y=start_y-level_sample.height
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
    def __init__(self, level, width, height, null_chars=['@'], wall_chars=['#'], start_direction=Direction.RIGHT, print_at_corners=False, change_direction_overrides=[]):
        self.level=level
        self.width=width
        self.height=height
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
        sample=self.get_sample_from_idx()
        print(f"Level sample at ({self.x_idx}, {self.y_idx}) on move step {self.move_iter}:")
        for row in sample:
            print(row)
        print("\n")

    #Gets a full level sample of the desired size from the top left corner
    def get_sample_from_idx(self, x=None, y=None):
        if x is None:
            x = self.x_idx
        if y is None:
            y = self.y_idx

        #Make sure the level sample is in bounds
        if x<0 or x<0:
            raise ValueError(f"X value ({x}) and Y value ({y}) all must be positive.")
        if (y + self.height)>len(self.level) or (x+self.width)>len(self.level[0]):
            raise ValueError(f"This level sample is out of bounds at the bottom or right, with height index {y+self.height}/{len(self.level)} and width index {x+self.width}/{len(self.level[0])}.")
        
        sample = []
        for row in self.level[y:y+self.height]:
            sample.append(row[x:x+self.width])
        
        return sample

    



        





if __name__ == "__main__":
    main()