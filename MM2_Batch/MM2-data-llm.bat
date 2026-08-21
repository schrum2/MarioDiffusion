@echo off
REM Usage: MM2-data-llm.bat [model] [seed]
REM Builds the MM2 dataset from ASCII levels with LLM captions (local Ollama).
REM Run MM2-extract.bat first to produce MM2_Data\ascii.
REM [model] is optional, defaults to qwen2.5:14b
REM [seed]  is optional, defaults to 0
cd ..

set MODEL=%1
if "%MODEL%"=="" set MODEL=qwen2.5:14b

set SEED=%2
if "%SEED%"=="" set SEED=0

REM Build a sliding-window dataset from the ASCII levels
python -m mm2pipeline_data dataset build --input MM2_Data\ascii --output_folder datasets\MM2_Levels-regular.json --tileset Game_MM2\mm2_tileset_we.json --sliding_window --stride 20

REM Caption every scene with a local Ollama LLM
python Game_MM2\MarioMaker_llm_captions.py --game MM2 --dataset datasets\MM2_Levels-regular.json --output datasets\MM2_LevelsAndCaptions-regular.json --backend ollama --model %MODEL% --grid-format tokens --num-captions 1 --caption-mode legacy

REM Tokenize MM2 data
python tokenizer.py save --json_file datasets\MM2_LevelsAndCaptions-regular.json --pkl_file datasets\MM2_Tokenizer-regular.pkl

REM Create validation captions
python create_random_test_captions.py --save_file datasets\MM2_RandomTest-regular.json --json datasets\MM2_LevelsAndCaptions-regular.json --seed %SEED% --game MM2

REM Split output files into train/val/test sets
python split_data.py --json_file datasets\MM2_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MM2
