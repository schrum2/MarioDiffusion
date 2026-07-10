@echo off
REM Usage: MM2_conditional-llm.bat [seed] [model]
REM Trains a text-conditional diffusion model on MM2 LLM captions using a pretrained text encoder.
REM Run MM2-data-llm.bat first to build and split the LLM-captioned dataset.
REM [seed]  is optional, defaults to 0
REM [model] should be "MiniLM", "GTE", "CLIP", or "T5", defaults to MiniLM
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set MODEL=%2
if /I "%MODEL%"=="" set MODEL=MiniLM
if /I "%MODEL%"=="MiniLM" set MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
if /I "%MODEL%"=="GTE" set MODEL_NAME=Alibaba-NLP/gte-large-en-v1.5
if /I "%MODEL%"=="CLIP" set MODEL_NAME=sentence-transformers/clip-ViT-L-14
if /I "%MODEL%"=="T5" set MODEL_NAME=google/t5-v1_1-base

set GAME=MM2
set NUM_TILES=68
set DIFF_OUTPUT=MM2-LLM-conditional-%MODEL%-%SEED%

set JSON_TRAIN=datasets\MM2_LevelsAndCaptions-regular-train.json
set JSON_VAL=datasets\MM2_LevelsAndCaptions-regular-validate.json
set JSON_TEST=datasets\MM2_LevelsAndCaptions-regular-test.json

REM Per-execution timing log: staged under timing_logs\ during the run, then moved
REM into the trained diffusion model's directory at the end.
set TIMING_LOG=timing_logs\%DIFF_OUTPUT%.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "MM2_conditional-llm pipeline start"

python train_diffusion.py --text_conditional --game %GAME% --json %JSON_TRAIN% --val_json %JSON_VAL% --pretrained_language_model "%MODEL_NAME%" --num_epochs 500 --output_dir "%DIFF_OUTPUT%" --seed %SEED%
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion training"

python run_diffusion.py --model_path "%DIFF_OUTPUT%" --num_samples 100 --text_conditional --save_as_json --output_dir "%DIFF_OUTPUT%-samples" --game %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "sampling"

python evaluate_caption_adherence.py --model_path "%DIFF_OUTPUT%" --json %JSON_TEST% --num_tiles %NUM_TILES% --game %GAME% --no_caption_score --save_as_json --output_dir samples-from-test-captions
python log_timestamp.py --log_file %TIMING_LOG% --event "caption adherence evaluation"

REM move the timing log into the trained model's directory
move /Y %TIMING_LOG% "%DIFF_OUTPUT%\pipeline_timing.jsonl"
