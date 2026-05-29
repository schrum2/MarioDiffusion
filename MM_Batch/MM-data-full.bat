@echo off
cd ..

if not exist "datasets" mkdir datasets

set default_out=datasets\MM_LevelsAndCaptions-full

:: Convert Mega Man raw level data to JSON
python create_megaman_json_data.py --game MM-Full

:: Generate captions for Mega Man
python MM_create_ascii_captions.py --output %default_out%-regular.json

:: Tokenize Mega Man data
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file datasets\MM_Tokenizer-full-regular.pkl

:: Create validation captions
python create_random_test_captions.py --save_file "datasets\\MM_RandomTest-full-regular.json" --json %default_out%-regular.json --seed 0 --game MM-Full

:: Split output files into train/val/test sets
python split_data.py --json %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game mm-full

:: Force add generated dataset files
git add --force datasets\MM_LevelsAndCaptions-full-regular.json
git add --force datasets\MM_Tokenizer-full-regular.pkl
git add --force datasets\MM_RandomTest-full-regular.json

git commit -m "Add MM-Full dataset files"

git push