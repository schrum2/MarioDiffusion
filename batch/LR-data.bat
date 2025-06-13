@echo off
cd ..

if not exist "datasets" mkdir datasets

:: Add LR-specific data processing commands here if needed
:: (This is a placeholder)

set default_out=datasets\LR_LevelsAndCaptions

:: Convert Lode Runner raw level data to JSON
python create_level_json_data.py --output "datasets\\LR_Levels.json" --levels "..\\TheVGLC\\Lode Runner\\Processed" --tileset "..\\TheVGLC\\Lode Runner\\Loderunner.json" --target_height 32 --target_width 32 --extra_tile .

:: Generate captions for Lode Runner
python LR_create_ascii_captions.py --dataset datasets\LR_Levels.json --output %default_out%-regular.json
python LR_create_ascii_captions.py --dataset datasets\LR_Levels.json --output %default_out%-absence.json --describe_absence

:: Tokenize Lode Runner data
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file datasets\LR_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out%-absence.json --pkl_file datasets\LR_Tokenizer-absence.pkl

:: Create validation captions for Lode Runner dataset, using the previously generated JSON files
python create_random_test_captions.py --save_file "datasets\\LR_RandomTest-regular.json" --json %default_out%-regular.json --seed 0 --game LR
python create_random_test_captions.py --save_file "datasets\\LR_RandomTest-absence.json" --json %default_out%-absence.json --seed 0 --describe_absence --game LR

:: Split output files into train/val/test sets
python split_data.py --json %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game loderunner
python split_data.py --json %default_out%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game loderunner
