cd ..

:: Merge SMB1 and SMB2 JSON datasets (assume SMB1 and SMB2 have already been processed)
python combine_data.py datasets\\SMB1AND2_Levels.json datasets\\SMB1_Levels.json datasets\\SMB2_Levels.json

:: Generate captions for SMB1AND2
default_out=datasets\\SMB1AND2_LevelsAndCaptions
python create_ascii_captions.py --dataset datasets\\SMB1AND2_Levels.json --output %default_out%-regular.json
python create_ascii_captions.py --dataset datasets\\SMB1AND2_Levels.json --output %default_out%-absence.json --describe_absence

:: Tokenize SMB1AND2 data
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file datasets\\SMB1AND2_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out%-absence.json --pkl_file datasets\\SMB1AND2_Tokenizer-absence.pkl

:: Create validation captions for SMB1AND2 dataset, using the previously generated JSON files
python create_validation_captions.py --save_file "datasets\\SMB1AND2_ValidationCaptions-regular.json" --json %default_out%-regular.json --seed 0
python create_validation_captions.py --save_file "datasets\\SMB1AND2_ValidationCaptions-absence.json" --json %default_out%-absence.json --seed 0 --describe_absence

:: Split output files into train/val/test sets
python split_data.py --json %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42

