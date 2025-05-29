@echo off
cd ..

if not exist "datasets" mkdir datasets

set default_out=datasets\SMB1_LevelsAndCaptions

:: Convert SMB1 raw level data to JSON
python create_level_json_data.py --output "datasets\\SMB1_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros\\Processed"

:: Generate captions for SMB1
python create_ascii_captions.py --dataset datasets\\SMB1_Levels.json --output %default_out%-regular.json
python create_ascii_captions.py --dataset datasets\\SMB1_Levels.json --output %default_out%-absence.json --describe_absence

:: Tokenize SMB1 data
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file datasets\\SMB1_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out%-absence.json --pkl_file datasets\\SMB1_Tokenizer-absence.pkl

:: Create validation captions for SMB1 dataset, using the previously generated JSON files
:: Added the --no_upside_down_pipes flag to SMB1 validation captions
python create_random_test_captions.py --save_file "datasets\\SMB1_RandomTest-regular.json" --json %default_out%-regular.json --seed 0 --no_upside_down_pipes
python create_random_test_captions.py --save_file "datasets\\SMB1_RandomTest-absence.json" --json %default_out%-absence.json --seed 0 --describe_absence --no_upside_down_pipes

:: Split output files into train/val/test sets
python split_data.py --json %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
