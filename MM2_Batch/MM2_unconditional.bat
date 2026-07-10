@echo off
REM Usage: MM2_unconditional.bat [seed]
REM Trains an unconditional diffusion model on MM2 levels.
REM Run MM2-data.bat first to build and split the dataset.
REM [seed] is optional, defaults to 0
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set GAME=MM2
set DIFF_OUTPUT=MM2-unconditional%SEED%

set JSON_TRAIN=datasets\MM2_LevelsAndCaptions-regular-train.json
set JSON_VAL=datasets\MM2_LevelsAndCaptions-regular-validate.json

REM Per-execution timing log: staged under timing_logs\ during the run, then moved
REM into the trained diffusion model's directory at the end.
set TIMING_LOG=timing_logs\%DIFF_OUTPUT%.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "MM2_unconditional pipeline start"

python train_diffusion.py --game %GAME% --augment --output_dir %DIFF_OUTPUT% --num_epochs 500 --json %JSON_TRAIN% --val_json %JSON_VAL% --seed %SEED%
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion training"

python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --save_as_json --output_dir "%DIFF_OUTPUT%-samples" --game %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "sampling"

REM move the timing log into the trained model's directory
move /Y %TIMING_LOG% "%DIFF_OUTPUT%\pipeline_timing.jsonl"
