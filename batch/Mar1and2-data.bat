@echo off
set WIDTH=%1
set WIDTH_SUFFIX=
if not "%WIDTH%"=="" set WIDTH_SUFFIX=_%WIDTH%

call SMB1-data.bat %WIDTH%
cd batch
call SMB2-data.bat %WIDTH%

set default_out=datasets\Mar1and2_LevelsAndCaptions%WIDTH_SUFFIX%

:: Merge SMB1 and SMB2 JSON datasets (assume SMB1 and SMB2 have already been processed)
python combine_data.py datasets\\Mar1and2_Levels%WIDTH_SUFFIX%.json datasets\\SMB1_Levels%WIDTH_SUFFIX%.json datasets\\SMB2_Levels%WIDTH_SUFFIX%.json

:: Generate captions for Mar1and2
python create_ascii_captions.py --dataset datasets\\Mar1and2_Levels%WIDTH_SUFFIX%.json --output %default_out%-regular.json
python create_ascii_captions.py --dataset datasets\\Mar1and2_Levels%WIDTH_SUFFIX%.json --output %default_out%-absence.json --describe_absence

:: Tokenize Mar1and2 data
python tokenizer.py save --json_file %default_out%-regular.json --pkl_file datasets\\Mar1and2_Tokenizer-regular%WIDTH_SUFFIX%.pkl
python tokenizer.py save --json_file %default_out%-absence.json --pkl_file datasets\\Mar1and2_Tokenizer-absence%WIDTH_SUFFIX%.pkl

:: Create validation captions for Mar1and2 dataset, using the previously generated JSON files
python create_random_test_captions.py --save_file "datasets\\Mar1and2_RandomTest-regular%WIDTH_SUFFIX%.json" --json %default_out%-regular.json --seed 0
python create_random_test_captions.py --save_file "datasets\\Mar1and2_RandomTest-absence%WIDTH_SUFFIX%.json" --json %default_out%-absence.json --seed 0 --describe_absence

:: Split output files into train/val/test sets
python split_data.py --json %default_out%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game mario
python split_data.py --json %default_out%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game mario
