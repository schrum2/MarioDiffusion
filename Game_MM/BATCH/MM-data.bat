@echo off
cd ..
if not exist "DATA" mkdir DATA
cd .. 

set simple_out=Game_MM/DATA/MM-Simple_LevelsAndCaptions
set full_out=Game_MM/DATA/MM-Full_LevelsAndCaptions

:: Convert Mega Man raw level data to JSON
python create_megaman_json_data.py --output Game_MM/DATA/MM-Full_Levels.json --direction_captions --no_filter --include_moving_ground
python create_megaman_json_data.py --output Game_MM/DATA/MM-Simple_Levels.json --direction_captions --no_filter --group_encodings 

:: Generate captions for Mega Man
python MM_create_ascii_captions.py --dataset Game_MM/DATA/MM-Full_Levels.json --tileset Game_MM/DATA/MM.json --output %full_out%-regular.json
python MM_create_ascii_captions.py --dataset Game_MM/DATA/MM-Simple_Levels.json --tileset Game_MM/DATA/MM-Simple-tileset.json --output %simple_out%-regular.json

:: Tokenize Mega Man data
python tokenizer.py save --json_file %full_out%-regular.json --pkl_file Game_MM/DATA/MM-Full_Tokenizer-regular.pkl
python tokenizer.py save --json_file %simple_out%-regular.json --pkl_file Game_MM/DATA/MM-Simple_Tokenizer-regular.pkl

:: Create validation captions
python create_random_test_captions.py --save_file "Game_MM/DATA/MM-Full_RandomTest-regular.json" --json %full_out%-regular.json --seed 0 --game MM-Full
python create_random_test_captions.py --save_file "Game_MM/DATA/MM-Simple_RandomTest-regular.json" --json %simple_out%-regular.json --seed 0 --game MM-Simple

:: Split output files into train/val/test sets
python split_data.py --json_file %full_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MM-Full
python split_data.py --json_file %simple_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game MM-Simple