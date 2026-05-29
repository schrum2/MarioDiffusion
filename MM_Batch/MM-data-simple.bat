@echo off
cd ..

if not exist "datasets" mkdir datasets

set default_out=datasets\MM_LevelsAndCaptions-simple

:: Convert Mega Man raw level data to JSON
python create_megaman_json_data.py --game MM-Simple

:: Generate captions for Mega Man
python MM_create_ascii_captions.py --output %default_out%-regular.json

:: Tokenize Mega Man data
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file datasets\MM_Tokenizer-simple-regular.pkl

:: Create validation captions
python create_random_test_captions.py --save_file "datasets\\MM_RandomTest-simple-regular.json" --json %default_out%-regular.json --seed 0 --game MM-Simple

:: Split output files into train/val/test sets
python split_data.py --json %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game mm-simple

:: Force add generated dataset files
git add --force datasets\MM_LevelsAndCaptions-simple-regular.json
git add --force datasets\MM_Tokenizer-simple-regular.pkl
git add --force datasets\MM_RandomTest-simple-regular.json

git commit -m "Add MM-Simple dataset files"

git push