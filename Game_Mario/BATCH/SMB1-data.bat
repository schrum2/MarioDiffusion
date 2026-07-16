@echo off
cd ..
if not exist "DATA" mkdir DATA
cd .. 

set WIDTH=%1
set WIDTH_SUFFIX=
set WIDTH_ARG=
if not "%WIDTH%"=="" (
    set WIDTH_SUFFIX=_%WIDTH%
    set WIDTH_ARG=--target_width %WIDTH%
)

set default_out=Game_Mario/DATA/SMB1%WIDTH_SUFFIX%_LevelsAndCaptions

:: Convert SMB1 raw level data to JSON
python create_level_json_data.py --output "Game_Mario/DATA/SMB1%WIDTH_SUFFIX%_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros\\Processed" %WIDTH_ARG%

:: Generate captions for SMB1
python create_ascii_captions.py --dataset Game_Mario/DATA/SMB1%WIDTH_SUFFIX%_Levels.json --output %default_out%-regular.json --exclude_upside_down_pipes
python create_ascii_captions.py --dataset Game_Mario/DATA/SMB1%WIDTH_SUFFIX%_Levels.json --output %default_out%-absence.json --exclude_upside_down_pipes --describe_absence

:: Tokenize SMB1 data
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file Game_Mario/DATA/SMB1%WIDTH_SUFFIX%_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out%-absence.json --pkl_file Game_Mario/DATA/SMB1%WIDTH_SUFFIX%_Tokenizer-absence.pkl

:: Create validation captions for SMB1 dataset, using the previously generated JSON files
:: Added the --no_upside_down_pipes flag to SMB1 validation captions
python create_random_test_captions.py --save_file "Game_Mario/DATA/SMB1%WIDTH_SUFFIX%_RandomTest-regular.json" --json %default_out%-regular.json --seed 0 --no_upside_down_pipes
python create_random_test_captions.py --save_file "Game_Mario/DATA/SMB1%WIDTH_SUFFIX%_RandomTest-absence.json" --json %default_out%-absence.json --seed 0 --describe_absence --no_upside_down_pipes 

:: Split output files into train/val/test sets
python split_data.py --json_file %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game mario
python split_data.py --json_file %default_out%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game mario
