@echo off
REM Usage: Mar1and2Mixed-data.bat
REM Builds the large EXPERIMENTAL mixed-width Mario dataset by generating the SMB1+SMB2
REM data at widths 16, 32, 64, and 128 and combining all of them into one dataset.

call Mar1and2-data.bat 16
cd Game_Mario/BATCH
call Mar1and2-data.bat 32
cd Game_Mario/BATCH
call Mar1and2-data.bat 64
cd Game_Mario/BATCH
call Mar1and2-data.bat 128

REM Combine the four widths into a single mixed dataset 
python combine_data.py Game_Mario/DATA/Mar1and2_16-32_LevelsAndCaptions-regular.json Game_Mario/DATA/Mar1and2_16_LevelsAndCaptions-regular.json Game_Mario/DATA/Mar1and2_32_LevelsAndCaptions-regular.json
python combine_data.py Game_Mario/DATA/Mar1and2_64-128_LevelsAndCaptions-regular.json Game_Mario/DATA/Mar1and2_64_LevelsAndCaptions-regular.json Game_Mario/DATA/Mar1and2_128_LevelsAndCaptions-regular.json
python combine_data.py Game_Mario/DATA/Mar1and2_16-32-64-128_LevelsAndCaptions-regular.json Game_Mario/DATA/Mar1and2_16-32_LevelsAndCaptions-regular.json Game_Mario/DATA/Mar1and2_64-128_LevelsAndCaptions-regular.json

REM Split the combined dataset into train/val/test sets
python split_data.py --json_file Game_Mario/DATA/Mar1and2_16-32-64-128_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game Mario

REM Build the tokenizer and random-test captions for the mixed dataset
python tokenizer.py save --json_file Game_Mario/DATA/Mar1and2_16-32-64-128_LevelsAndCaptions-regular.json --pkl_file Game_Mario/DATA/Mar1and2Mixed_Tokenizer-regular.pkl
python create_random_test_captions.py --save_file "Game_Mario/DATA/Mar1and2_16-32-64-128_RandomTest-regular.json" --json Game_Mario/DATA/Mar1and2_16-32-64-128_LevelsAndCaptions-regular.json --seed 0
