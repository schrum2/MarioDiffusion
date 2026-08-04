@echo off
cd ..
cd ..

python create_tile_level_json_data.py --tileset "Game_MM/MM.json" --levels "..\TheVGLC\MegaMan\Enhanced" --output "Game_MM/DATA/MM-Full_3x3_tiles.json" --tile_size 3 
python create_tile_level_json_data.py --tileset "Game_MM/MM-Simple-tileset.json" --levels "..\TheVGLC\MegaMan\Enhanced" --output "Game_MM/DATA/MM-Simple_3x3_tiles.json" --tile_size 3 --char_map "Game_MM/MM-VGLC-to-simple.json"
