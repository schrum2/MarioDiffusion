cd ..

if not exist "datasets" mkdir datasets

set default_out=datasets\SMB2_LevelsAndCaptions

:: Convert SMB2 raw level data to JSON
python create_level_json_data.py --output "datasets\\SMB2_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros 2 (Japan)\\Processed"

:: Generate captions for SMB2
python create_ascii_captions.py --dataset datasets\\SMB2_Levels.json --output %default_out%-regular.json
python create_ascii_captions.py --dataset datasets\\SMB2_Levels.json --output %default_out%-absence.json --describe_absence

:: Tokenized SMB2 data for each dataset into pickle files
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file datasets\\SMB2_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out%-absence.json --pkl_file datasets\\SMB2_Tokenizer-absence.pkl

:: Create validation captions for SMB2 dataset, using the previously generated JSON files
python create_validation_captions.py --save_file "datasets\\SMB2_ValidationCaptions-regular.json" --json %default_out%-regular.json --seed 0
python create_validation_captions.py --save_file "datasets\\SMB2_ValidationCaptions-absence.json" --json %default_out%-absence.json --seed 0 --describe_absence

:: Split output files into train/val/test sets
python split_data.py --json %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42