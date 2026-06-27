@echo off


cd ..


:: Convert Mega Man raw level data to JSON
python create_megaman_json_data.py --output datasets\\MM-full_Levels.json
python create_megaman_json_data.py --output datasets\\MM-simple_Levels.json --group_encodings

:: Generate captions for Mega Man
python MM_create_ascii_captions.py --dataset datasets\\MM-full_Levels.json --tileset datasets\\MM-full_tileset.json --output datasets\\MM-full_LevelsAndCaptions-regular.json
python MM_create_ascii_captions.py --dataset datasets\\MM-simple_Levels.json --tileset datasets\\MM-simple_tileset.json --output datasets\\MM-simple_LevelsAndCaptions-regular.json

:: Tokenize Mega Man data
python tokenizer.py save --json_file datasets\\MM-full_LevelsAndCaptions-regular.json --pkl_file datasets\MM-full_Tokenizer-regular.pkl
python tokenizer.py save --json_file datasets\\MM-simple_LevelsAndCaptions-regular.json --pkl_file datasets\MM-simple_Tokenizer-regular.pkl

:: Create validation captions
python create_random_test_captions.py --save_file "datasets\\MM-full_RandomTest-regular.json" --json datasets\\MM-full_LevelsAndCaptions-regular.json --seed 0 --game MM-Full
python create_random_test_captions.py --save_file "datasets\\MM-simple_RandomTest-regular.json" --json datasets\\MM-simple_LevelsAndCaptions-regular.json --seed 0 --game MM-Simple

:: Split output files into train/val/test sets
python split_data.py --json_file datasets\\MM-full_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MM-Full
python split_data.py --json_file datasets\\MM-simple_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MM-Simple