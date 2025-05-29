cd ..

if not exist "datasets" mkdir datasets

set default_out=datasets\SML_LevelsAndCaptions

:: Convert SML raw level data to JSON
python create_level_json_data.py --output "datasets\\SML_Levels.json" --levels "..\\TheVGLC\\Super Mario Land\\Processed"

:: Generate captions for SML
python create_ascii_captions.py --dataset datasets\\SML_Levels.json --output %default_out%-regular.json
python create_ascii_captions.py --dataset datasets\\SML_Levels.json --output %default_out%-absence.json --describe_absence

:: Tokenize SML data
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file datasets\\SML_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out%-absence.json --pkl_file datasets\\SML_Tokenizer-absence.pkl

:: Create validation captions for SML dataset, using the previously generated JSON files
python create_random_test_captions.py --save_file "datasets\\SML_RandomTest-regular.json" --json %default_out%-regular.json --seed 0
python create_random_test_captions.py --save_file "datasets\\SML_RandomTest-absence.json" --json %default_out%-absence.json --seed 0 --describe_absence

:: Split output files into train/val/test sets
python split_data.py --json %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
