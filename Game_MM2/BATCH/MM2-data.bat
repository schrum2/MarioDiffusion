@echo off
REM Usage: MM2-data.bat [seed]
REM Builds the MM2 dataset from ASCII levels with deterministic tile-presence captions.
REM Run MM2-extract.bat first to produce MM2_Data\ascii.
REM [seed] is optional, defaults to 0
cd ..
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

REM Build a sliding-window dataset from the ASCII levels
python -m mm2pipeline_data dataset build --input Game_MM2\LEVELS\ascii --output_folder Game_MM2\DATA\MM2_Levels-regular.json --tileset Game_MM2\mm2_tileset_we.json --sliding_window --stride 20

REM Generate deterministic captions for MM2
python Game_MM2\MarioMaker_create_ascii_captions.py --dataset Game_MM2\DATA\MM2_Levels-regular.json --tileset Game_MM2\mm2_tileset_we.json --output Game_MM2\DATA\MM2_LevelsAndCaptions-regular.json

REM Tokenize MM2 data
python Game_MM2\tokenizer.py save --json_file Game_MM2\DATA\MM2_LevelsAndCaptions-regular.json --pkl_file Game_MM2\DATA\MM2_Tokenizer-regular.pkl

REM Create validation captions
python Game_MM2\create_random_test_captions.py --save_file Game_MM2\DATA\MM2_RandomTest-regular.json --json Game_MM2\DATA\MM2_LevelsAndCaptions-regular.json --seed %SEED% --game MM2

REM Split output files into train/val/test sets
python split_data.py --json_file Game_MM2\DATA\MM2_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MM2
