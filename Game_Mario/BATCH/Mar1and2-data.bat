@echo off
set WIDTH=%1
set WIDTH_SUFFIX=
if not "%WIDTH%"=="" set WIDTH_SUFFIX=_%WIDTH%

call SMB1-data.bat %WIDTH%
cd Game_Mario/BATCH
call SMB2-data.bat %WIDTH%

set default_out=Game_Mario/DATA/Mar1and2%WIDTH_SUFFIX%_LevelsAndCaptions

:: Merge SMB1 and SMB2 JSON datasets (assume SMB1 and SMB2 have already been processed)
python combine_data.py Game_Mario/DATA/Mar1and2%WIDTH_SUFFIX%_Levels.json Game_Mario/DATA/SMB1%WIDTH_SUFFIX%_Levels.json Game_Mario/DATA/SMB2%WIDTH_SUFFIX%_Levels.json

:: Generate captions for Mar1and2
python create_ascii_captions.py --dataset Game_Mario/DATA/Mar1and2%WIDTH_SUFFIX%_Levels.json --output %default_out%-regular.json
python create_ascii_captions.py --dataset Game_Mario/DATA/Mar1and2%WIDTH_SUFFIX%_Levels.json --output %default_out%-absence.json --describe_absence

:: Tokenize Mar1and2 data
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file Game_Mario/DATA/Mar1and2%WIDTH_SUFFIX%_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out%-absence.json --pkl_file Game_Mario/DATA/Mar1and2%WIDTH_SUFFIX%_Tokenizer-absence.pkl

:: Create validation captions for Mar1and2 dataset, using the previously generated JSON files
python create_random_test_captions.py --save_file "Game_Mario/DATA/Mar1and2%WIDTH_SUFFIX%_RandomTest-regular.json" --json %default_out%-regular.json --seed 0
python create_random_test_captions.py --save_file "Game_Mario/DATA/Mar1and2%WIDTH_SUFFIX%_RandomTest-absence.json" --json %default_out%-absence.json --seed 0 --describe_absence

:: Split output files into train/val/test sets
python split_data.py --json_file %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game Mario
python split_data.py --json_file %default_out%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game Mario
