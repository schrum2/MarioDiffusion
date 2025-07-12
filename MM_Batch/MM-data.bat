@echo off
cd ..


:: Convert Mega Man raw level data to JSON
python create_megaman_json_data.py --output datasets\\MM_Levels_Full.json
python create_megaman_json_data.py --output datasets\\MM_Levels_Simple.json --group_encodings

:: Generate captions for Mega Man
python MM_create_ascii_captions.py --dataset datasets\\MM_Levels_Full.json --tileset datasets\\MM.json --output datasets\\MM_LevelsAndCaptions-full-regular.json
python MM_create_ascii_captions.py --dataset datasets\\MM_Levels_Simple.json --tileset datasets\\MM_Simple_Tileset.json --output datasets\\MM_LevelsAndCaptions-simple-regular.json

:: Tokenize Mega Man data
python tokenizer.py save --json datasets\\MM_LevelsAndCaptions-full-regular.json --pkl_file datasets\MM_Tokenizer-full-regular.pkl
python tokenizer.py save --json datasets\\MM_LevelsAndCaptions-simple-regular.json --pkl_file datasets\MM_Tokenizer-simple-regular.pkl

:: Validation captions making, this is not compatable yet
REM python create_random_test_captions.py --save_file "datasets\\MM_RandomTest-simple-regular.json" --json datasets\\MM_LevelsAndCaptions-simple-regular.json --seed 0 --game MM-Simple
REM python create_random_test_captions.py --save_file "datasets\\LR_RandomTest-absence.json" --json %default_out%-absence.json --seed 0 --describe_absence --game LR

:: Split output files into train/val/test sets, also not done
REM python split_data.py --json %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game loderunner
REM python split_data.py --json %default_out%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game loderunner
