@echo off
cd ..
cd ..

python create_tile_level_json_data.py --tileset "Game_MM-Simple/MM-Simple-tileset.json" --levels "..\TheVGLC\MegaMan\Enhanced" --output "Game_MM-Simple/DATA/MM-Simple_3x3_tiles.json" --tile_size 3 --char_map "Game_MM-Simple/MM-VGLC-to-simple.json"
