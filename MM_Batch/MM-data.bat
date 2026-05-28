@echo off
REM MM-data.bat
REM Creates Mega Man datasets, captions, tokenizers, train/val/test splits, and random test captions.
cd ..

:: Convert Mega Man raw level data to JSON
python create_megaman_json_data.py --output datasets\\MM_Levels_Full.json
python create_megaman_json_data.py --output datasets\\MM_Levels_Simple.json --group_encodings

:: Generate captions for Mega Man
python MM_create_ascii_captions.py --dataset datasets\\MM_Levels_Full.json --tileset datasets\\MM.json --output datasets\\MM_LevelsAndCaptions-full-regular.json
python MM_create_ascii_captions.py --dataset datasets\\MM_Levels_Simple.json --tileset datasets\\MM_Simple_Tileset.json --output datasets\\MM_LevelsAndCaptions-simple-regular.json

:: Tokenize Mega Man data
python tokenizer.py save --json datasets\\MM_LevelsAndCaptions-full-regular.json --pkl_file datasets\\MM_Tokenizer-full-regular.pkl
python tokenizer.py save --json datasets\\MM_LevelsAndCaptions-simple-regular.json --pkl_file datasets\\MM_Tokenizer-simple-regular.pkl

:: Create random test captions for MM-Simple (sampled from dataset, no grammar generator)
python create_random_test_captions.py --save_file "datasets\\MM_RandomTest-simple-regular.json" --json datasets\\MM_LevelsAndCaptions-simple-regular.json --seed 0 --game MM-Simple

:: Split output files into train/val/test sets
python split_data.py --json datasets\\MM_LevelsAndCaptions-simple-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game megaman
python split_data.py --json datasets\\MM_LevelsAndCaptions-full-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game megaman