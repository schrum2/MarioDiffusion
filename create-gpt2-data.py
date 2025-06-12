import create_level_json_data as json_data_helper
import json
import os


#Takes in a directory path to a bunch of levels
#Removes pathing from levels
#Splits levels into small blocks of size 16X16 (needs to pad for this)
    #Use create_level_json_data for this
#converts to JSON data (use create_level_json_data for this)

level_dir = "SMB1-gpt-levels/levels" #Define as args parameter later
path_char='x'
pad_char='-'
tileset_path='..\\TheVGLC\\Super Mario Bros\\smb.json'
def main():

    level_files = [f for f in os.listdir(level_dir) if f.endswith(".txt")]
    #Loads all levels as a list of lists of strings
    levels_as_lists = []
    for filename in level_files:
        with open(os.path.join(level_dir, filename), "r") as f:
            lines = [line.rstrip("\n") for line in f]
            levels_as_lists.append(lines)

    #needed for this call to work
    tile_to_id = json_data_helper.load_tileset(tileset_path, extra_tile=pad_char)
    for level in levels_as_lists:
        for row in level:
            row = row.replace(path_char, pad_char)
        
        split_level = pad_and_split(long_level=level, tile_to_id=tile_to_id, target_height=16, target_width=16, pad_tile=pad_char, tileset_path=tileset_path)
        for lvl in split_level:
            print("\n\n\n\nNew Level: ")
            for row in lvl:
                for char in row:
                    print(char, end=" ")
                print(" ")


def pad_and_split(long_level, tile_to_id, target_height, target_width, pad_tile, tileset_path):
    # Platformer: original sliding window
    height = len(long_level)
    width = max(len(row) for row in long_level)
    pad_rows = target_height - height
    padded_level = [pad_tile * width] * pad_rows + long_level

    # Special case for SMB2 upside down pipes that extend into the sky
    _, _, _, tile_descriptors = json_data_helper.extract_tileset(tileset_path) # partially duplicates the work of load_tileset, except for the extra_tile
    for i in range(width): # Scan top row
        descriptors = tile_descriptors.get(long_level[0][i], [])
        if "pipe" in descriptors:
            # If the top row has a pipe, extend it upwards
            for j in range(pad_rows):
                row_list = list(padded_level[j])
                row_list[i] = long_level[0][i]
                padded_level[j] = ''.join(row_list)

    padded_level = [row.ljust(target_width, pad_tile) for row in padded_level]
    samples = []
    for x in range(width//target_width):
        sample = []
        for y in range(target_height):
            window_row = padded_level[y][x*target_width:(x+1)*target_width]
            sample.append([tile_to_id.get(c, tile_to_id[pad_tile]) for c in window_row])
        samples.append(sample)
    return samples



if __name__ == "__main__":
    main()