import json
import argparse
from pathlib import Path
import util.common_settings as common_settings
from captions.util import extract_tileset
from create_level_json_data import load_levels
from enum import Enum
import os
import sys

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
        #print(tile_chars)
        #print(tile_descriptors)

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

        #Create a new tileset to match these tiles
        output = os.path.join(new_tileset_dir, "MM_Simple_Tileset.json")
        tile_dict = {tile: list(tile_descriptors.get(tile)) for tile in tile_to_id}
        tile_dict = {"tiles" : tile_dict}
        
        with open(output, 'w') as f:
            json.dump(tile_dict, f)

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

    # These control the OUTPUT scene size only. Navigation through the level always
    # uses the MegaMan screen size (common_settings.MEGAMAN_WIDTH x MEGAMAN_HEIGHT);
    # the path follower would stall if it had to fit a larger window between the null
    # regions that border every corridor. Output defaults to a 16x16 square so the
    # default dataset is byte-identical to the previous behaviour.
    parser.add_argument('--target_height', type=int, default=common_settings.MEGAMAN_WIDTH, help='Output scene height (e.g., 16 or 32). Navigation still uses the screen height.')
    parser.add_argument('--target_width', type=int, default=common_settings.MEGAMAN_WIDTH, help='Output scene width (e.g., 16 or 32). Navigation still uses the screen width.')

    # For scenes taller than the screen, the rows above the navigation window can either
    # be filled with the real level content above the screen (so a vertical shaft fills the
    # whole scene) or with null "sky" padding. Faithful filling turns on automatically once
    # the output is taller than the default square; this flag forces it on at any size.
    # Default (off) keeps the original 16x16 dataset byte-identical.
    parser.add_argument('--faithful_vertical', action='store_true', help='Fill the rows above the navigation window with real level content instead of null padding (auto-enabled when --target_height exceeds the default square).')

    parser.add_argument('--group_encodings', action='store_true', help='Group the tile encodings by type to reduce the total number')


    # After all scenes are generated, run the A* traversability check on them and drop
    # the ones MegaMan cannot complete, so the written dataset only contains beatable slices.
    parser.add_argument('--traversable_only', action='store_true', help='Filter out un-traversable scenes (via the A* check) before writing the dataset')
    parser.add_argument('--budget', type=int, default=100000, help='A* state-expansion budget per scene used by --traversable_only (higher = more thorough, slower)')


    return parser.parse_args()


def filter_traversable(all_samples, id_to_char, tile_descriptors, budget=100000):
    """Drop the samples whose scene the A* check can't traverse and report the survival rate.

    Removes by descending index so earlier indices stay valid while deleting. id_to_char and
    tile_descriptors must match the encoding the samples were written with (the same ones used
    to encode them above), so the A* check decodes each tile correctly."""
    # astar/ lives beside this file and manages its own internal imports off sys.path.
    astar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astar")
    if astar_dir not in sys.path:
        sys.path.insert(0, astar_dir)
    from astar_traversability_check import untraversable_indices

    scenes = [s["sample"] for s in all_samples]
    bad_indices = untraversable_indices(scenes, "MM", id_to_char, tile_descriptors, budget=budget)

    for idx in sorted(bad_indices, reverse=True):
        del all_samples[idx]

    total = len(scenes)
    remaining = len(all_samples)
    pct = (100.0 * remaining / total) if total else 0.0
    print(f"Traversability filter: removed {len(bad_indices)}/{total} un-traversable scenes; "
          f"{remaining} levels remain ({pct:.1f}% of the dataset).")
    return all_samples


def main():

    args = parse_args()



    levels = load_levels(args.levels)
    _, id_to_char, tile_to_id, tile_descriptors = extract_tileset(args.tileset)
    null_chars = [key for key, value in tile_descriptors.items() if 'null' in value]
    wall_chars = [key for key, value in tile_descriptors.items() if (('solid' in value) and ('penetrable' not in value))]
    #print(null_chars)
    #print(wall_chars)
    
    if args.group_encodings:
        # Grouping remaps the ids, so refresh id_to_char to the grouped encoding (each id ->
        # its representative char) to keep the A* filter's decoding in sync with the samples.
        tile_to_id, id_to_char = create_tile_to_id(args.tileset, tile_descriptors, new_tileset_dir=os.path.dirname(args.output))
    
    #We literally only need level overrides for 1-7, every other level parses as expected
    overrides_1_7 = [120, 121, 122, 123, 182] #Needed to avoid an early turn leading to a split path, and to prevent the level from turning back around to go back to the start

    #Navigation always uses the MegaMan screen size; the output size is what the user requested.
    nav_width = common_settings.MEGAMAN_WIDTH
    nav_height = common_settings.MEGAMAN_HEIGHT

    #Faithfully fill the rows above the screen with real content for taller-than-default
    #scenes (or when explicitly requested); the default square keeps the legacy null padding.
    faithful_vertical = args.faithful_vertical or (args.target_height > nav_width)

    all_samples = []
    for i in range(len(levels)):
        if i==7: #We need to do some slight overrides on 1-7 to make the level functional
            samples, json_caption_data=parse_level(tile_to_id, levels[i], nav_width, nav_height, null_chars, wall_chars, out_width=args.target_width, out_height=args.target_height, faithful_vertical=faithful_vertical, print_at_corners=False, change_direction_overrides=overrides_1_7)
        else:
            samples, json_caption_data=parse_level(tile_to_id, levels[i], nav_width, nav_height, null_chars, wall_chars, out_width=args.target_width, out_height=args.target_height, faithful_vertical=faithful_vertical, print_at_corners=False)
        
        #We do this so each level scene is encoded together, not grouped by level
        for sample, json_data in zip(samples, json_caption_data):
            all_samples.append({
                "sample" :sample,
                "data": json_data
                })    
    
    #Optionally drop scenes MegaMan can't actually complete before writing the dataset
    if args.traversable_only:
        all_samples = filter_traversable(all_samples, id_to_char, tile_descriptors, budget=args.budget)

    #Move everything to a json file
    output = args.output
    with open(output, 'w') as f:
        json.dump(all_samples, f, indent=2)

    


#Parses through one complete level
#width/height are the NAVIGATION (screen) dimensions; out_width/out_height are the
#output scene dimensions (default to a square of side `width` to match the old behaviour).
def parse_level(tile_to_id, level, width, height, null_chars=['@'], wall_chars=['#'], out_width=None, out_height=None, faithful_vertical=False, start_direction=Direction.RIGHT, print_at_corners=False, change_direction_overrides=[]):
    level_sample=LevelSample(level, width, height, null_chars, wall_chars, out_width=out_width, out_height=out_height, faithful_vertical=faithful_vertical, start_direction=start_direction, print_at_corners=print_at_corners, change_direction_overrides=change_direction_overrides)
    samples = []
    json_caption_data = []

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
    json_caption_data.append(get_json_caption_data(level_sample, prev_direction, current_direction))

    while moving:
        prev_direction=current_direction

        moving=level_sample.move_step()
        
        if not moving: 
            break

        samples.append(level_sample.get_sample_from_idx())

        current_direction = level_sample.direction

        json_caption_data.append(get_json_caption_data(level_sample, prev_direction, current_direction))


    encoded_samples = []
    for sample in samples:
        encoded_sample = []
        for row in sample:
            #Index directly (not .get) so a character missing from the tileset fails
            #loudly here instead of silently encoding to None and breaking captioning.
            encoded_sample.append([tile_to_id[c] for c in row])
        encoded_samples.append(encoded_sample)
    #level_sample.print_sample()
    return encoded_samples, json_caption_data



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
            start_y=max(0, start_y-level_sample.height) #This is needed because we expect a top left index, not a bottom left
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
        self.width=width    #navigation window width (screen size)
        self.height=height  #navigation window height (screen size)
        #Output scene size. Defaults to a square of side `width`, which reproduces the
        #original "pad to square" behaviour exactly (width x width, e.g. 16x16).
        self.out_width = out_width if out_width is not None else width
        self.out_height = out_height if out_height is not None else width
        #When True, rows above the navigation window show real level content; when False
        #they are null padding (preserves the legacy default output byte-for-byte).
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

    #Gets a full level sample of the desired output size, anchored on the navigation window
    def get_sample_from_idx(self, x=None, y=None, pad_sample=True):
        if x is None:
            x = self.x_idx
        if y is None:
            y = self.y_idx

        #The navigation window itself must stay in bounds
        if x<0 or y<0:
            raise ValueError(f"X value ({x}) and Y value ({y}) all must be positive.")
        if (y + self.height)>len(self.level) or (x+self.width)>len(self.level[0]):
            raise ValueError(f"This level sample is out of bounds at the bottom or right, with height index {y+self.height}/{len(self.level)} and width index {x+self.width}/{len(self.level[0])}.")

        if not pad_sample:
            #Raw navigation window (used for debug printing)
            return [row[x:x+self.width] for row in self.level[y:y+self.height]]

        null = self.null_chars[0]
        level_height = len(self.level)
        level_width = len(self.level[0])

        #The output scene is a slice of the level, sized to the requested output dimensions
        #and anchored on the navigation window:
        # - The nav window is bottom-anchored in the output (its floor row stays at the
        #   bottom, matching the platformer "ground at the bottom" convention) and
        #   centered horizontally.
        # - The navigation window itself, and everything horizontally beside it, is filled
        #   with the real level tile at that position (corridors continue sideways).
        # - The rows ABOVE the navigation window are real level content when
        #   faithful_vertical is on (so a tall vertical shaft fills all 32 rows), or null
        #   "sky" padding when off (this reproduces the legacy default output exactly).
        # - Null is only used where the slice falls outside the level grid.
        col_offset = (self.out_width - self.width) // 2
        row_offset = self.out_height - self.height  #output rows above the nav window
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
                    out_row.append(null)  #legacy null "sky" padding above the screen
                elif 0 <= ly < level_height and 0 <= lx < level_width:
                    out_row.append(self.level[ly][lx])
                else:
                    out_row.append(null)
            sample.append(out_row)

        return sample

    



        





if __name__ == "__main__":
    main()