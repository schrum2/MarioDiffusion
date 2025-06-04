@echo off
call SMB1-data.bat
cd batch
call SMB2-data.bat

set default_out=datasets\Mario_LevelsAndCaptions

:: Merge SMB1, SMB2, and SML JSON datasets (assume previous batch files have already been run)
python combine_data.py datasets\\Mario_Levels.json datasets\\SMB1_Levels.json datasets\\SMB2_Levels.json datasets\\SML_Levels.json

:: Generate captions for Mario
python create_ascii_captions.py --dataset datasets\\Mario_Levels.json --output %default_out%-regular.json
python create_ascii_captions.py --dataset datasets\\Mario_Levels.json --output %default_out%-absence.json --describe_absence

:: Tokenize Mario data
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file datasets\\Mario_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out%-absence.json --pkl_file datasets\\Mario_Tokenizer-absence.pkl

:: Create validation captions for Mario dataset, using the previously generated JSON files
python create_random_test_captions.py --save_file "datasets\\Mario_RandomTest-regular.json" --json %default_out%-regular.json --seed 0
python create_random_test_captions.py --save_file "datasets\\Mario_RandomTest-absence.json" --json %default_out%-absence.json --seed 0 --describe_absence

:: Split output files into train/val/test sets
python split_data.py --json %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game mario
python split_data.py --json %default_out%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game mario
