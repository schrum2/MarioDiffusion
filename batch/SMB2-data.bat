@echo off
cd ..

if not exist "datasets" mkdir datasetset WIDTH=%1
set WIDTH_SUFFIX=
set WIDTH_ARG=
if not "%WIDTH%"=="" (
    set WIDTH_SUFFIX=_%WIDTH%
    set WIDTH_ARG=--target_width %WIDTH%
)

set default_out=datasets\SMB2_LevelsAndCaptions%WIDTH_SUFFIX%

:: Convert SMB2 raw level data to JSON
python create_level_json_data.py --output "datasets\\SMB2_Levels%WIDTH_SUFFIX%.json" --levels "..\\TheVGLC\\Super Mario Bros 2 (Japan)\\Processed" %WIDTH_ARG%

:: Generate captions for SMB2
python create_ascii_captions.py --dataset datasets\\SMB2_Levels%WIDTH_SUFFIX%.json --output %default_out%-regular.json
python create_ascii_captions.py --dataset datasets\\SMB2_Levels%WIDTH_SUFFIX%.json --output %default_out%-absence.json --describe_absence

:: Tokenized SMB2 data for each dataset into pickle files
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file datasets\\SMB2_Tokenizer-regular%WIDTH_SUFFIX%.pkl
python tokenizer.py save --json_file %default_out%-absence.json --pkl_file datasets\\SMB2_Tokenizer-absence%WIDTH_SUFFIX%.pkl

:: Create validation captions for SMB2 dataset, using the previously generated JSON files
python create_random_test_captions.py --save_file "datasets\\SMB2_RandomTest-regular%WIDTH_SUFFIX%.json" --json %default_out%-regular.json --seed 0
python create_random_test_captions.py --save_file "datasets\\SMB2_RandomTest-absence%WIDTH_SUFFIX%.json" --json %default_out%-absence.json --seed 0 --describe_absence
 --describe_absence

:: Split output files into train/val/test sets
python split_data.py --json %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game mario
python split_data.py --json %default_out%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game mario
