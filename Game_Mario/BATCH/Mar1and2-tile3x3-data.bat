@echo off
cd ..
cd ..

python create_tile_level_json_data.py --output "Game_Mario/DATA/SMB1_3x3_tiles.json" --tile_size 3 --levels "..\TheVGLC\Super Mario Bros\Processed"
python create_tile_level_json_data.py --output "Game_Mario/DATA/SMB2_3x3_tiles.json" --tile_size 3 --levels "..\TheVGLC\Super Mario Bros 2 (Japan)\Processed"
python combine_data.py "Game_Mario/DATA/Mar1and2_3x3_tiles.json" "Game_Mario/DATA/SMB1_3x3_tiles.json" "Game_Mario/DATA/SMB2_3x3_tiles.json"