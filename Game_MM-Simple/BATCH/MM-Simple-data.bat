@echo off
cd ..
if not exist "DATA" mkdir DATA
cd .. 

set default_out=Game_MM-Simple/DATA/MM-Simple_LevelsAndCaptions

:: Convert Mega Man raw level data to JSON
python create_megaman_json_data.py --output Game_MM-Simple/DATA/MM-Simple_Levels.json --group_encodings --direction_captions --no_filter

:: Generate captions for Mega Man
python MM_create_ascii_captions.py --dataset Game_MM-Simple/DATA/MM-Simple_Levels.json --tileset Game_MM-Simple/MM-simple-tileset.json --output %default_out%-regular.json

:: Tokenize Mega Man data
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file Game_MM-Simple/DATA/MM-Simple_Tokenizer-regular.pkl

:: Create validation captions
python create_random_test_captions.py --save_file "Game_MM-Simple/DATA/MM-Simple_RandomTest-regular.json" --json %default_out%-regular.json --seed 0 --game MM-Simple

:: Split output files into train/val/test sets
python split_data.py --json_file %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game MM-Simple