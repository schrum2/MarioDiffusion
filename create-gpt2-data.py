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
        
        level = json_data_helper.pad_and_sample(level=level, tile_to_id=tile_to_id, target_height=16, target_width=16, extra_tile=pad_char)
        print(level)



if __name__ == "__main__":
    main()