@echo off
REM Usage: MM2_conditional.bat [seed]
REM Trains a text-conditional diffusion model on MM2 levels using our own MLM text encoder.
REM Run MM2-data.bat first to build and split the dataset.
REM [seed] is optional, defaults to 0
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set GAME=MM2
set NUM_TILES=68

set JSON_TRAIN=datasets\MM2_LevelsAndCaptions-regular-train.json
set JSON_VAL=datasets\MM2_LevelsAndCaptions-regular-validate.json
set JSON_TEST=datasets\MM2_LevelsAndCaptions-regular-test.json
set JSON_RANDOM=datasets\MM2_RandomTest-regular.json
set PKL=datasets\MM2_Tokenizer-regular.pkl
set MLM_OUTPUT=MM2-MLM-regular%SEED%
set DIFF_OUTPUT=MM2-conditional-regular%SEED%

REM Per-execution timing log: staged under timing_logs\ during the run, then moved
REM into the trained diffusion model's directory at the end.
set TIMING_LOG=timing_logs\%DIFF_OUTPUT%.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "MM2_conditional pipeline start"

python train_mlm.py --epochs 300 --save_checkpoints --json %JSON_TRAIN% --val_json %JSON_VAL% --test_json %JSON_TEST% --pkl %PKL% --output_dir %MLM_OUTPUT% --seed %SEED%
python log_timestamp.py --log_file %TIMING_LOG% --event "MLM training"

python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir %DIFF_OUTPUT% --num_epochs 500 --json %JSON_TRAIN% --val_json %JSON_VAL% --pkl %PKL% --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% --game %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion training"

REM Generate 100 conditional samples from the trained model
python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --text_conditional --save_as_json --output_dir "%DIFF_OUTPUT%-samples" --game %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "sampling"

REM Evaluate caption adherence on the test and random caption sets
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json %JSON_TEST% --compare_checkpoints --num_tiles %NUM_TILES% --game %GAME%
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json %JSON_RANDOM% --output_dir "%DIFF_OUTPUT%-caption-adherence-random" --num_tiles %NUM_TILES% --game %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "caption adherence evaluation"

REM Evaluate MM2-specific metrics
python -m MM2_Files.evaluate_mm2_metrics --model_path %DIFF_OUTPUT% --game %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "MM2 metrics evaluation"

REM move the timing log into the trained model's directory
move /Y %TIMING_LOG% "%DIFF_OUTPUT%\pipeline_timing.jsonl"
