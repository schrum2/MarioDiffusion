@echo off
REM Usage: MM_conditional.bat [simple|full] [seed]
REM Defaults to "simple" and seed 0 if no arguments provided
cd ..

set VARIANT=%1
if "%VARIANT%"=="" set VARIANT=simple

set SEED=%2
if "%SEED%"=="" set SEED=0

if "%VARIANT%"=="full" (
    set GAME=MM-Full
    set DATASET_INFIX=full
    set TILESET=datasets\MM.json
    set RAW_JSON=datasets\MM-full_Levels.json
    set NUM_TILES=41
) else (
    set GAME=MM-Simple
    set DATASET_INFIX=simple
    set TILESET=Game_MM-Simple/MM-simple-tileset.json
    set RAW_JSON=datasets\MM-simple_Levels.json
    set NUM_TILES=13
)

set JSON_TRAIN=datasets\MM-%DATASET_INFIX%_LevelsAndCaptions-regular.json
set JSON_TEST=datasets\MM-%DATASET_INFIX%_LevelsAndCaptions-regular-test.json
set JSON_RANDOM=datasets\MM-%DATASET_INFIX%_RandomTest-regular.json
set PKL=datasets\MM-%DATASET_INFIX%_Tokenizer-regular.pkl
set MLM_OUTPUT=MM-%DATASET_INFIX%-MLM-regular%SEED%
set DIFF_OUTPUT=MM-%DATASET_INFIX%-conditional-regular%SEED%

REM Per-execution timing log: staged under timing_logs\ during the run, then moved
REM into the trained diffusion model's directory at the end.
set TIMING_LOG=timing_logs\%DIFF_OUTPUT%.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "MM_conditional pipeline start"

REM Run MM-data.bat first to generate the necessary JSON and PKL files

python train_mlm.py --epochs 300 --save_checkpoints --json %JSON_TRAIN% --pkl %PKL% --output_dir %MLM_OUTPUT% --seed %SEED%
python log_timestamp.py --log_file %TIMING_LOG% --event "MLM training"
python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir %DIFF_OUTPUT% --num_epochs 500 --json %JSON_TRAIN% --pkl %PKL% --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% --game %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion training"

REM call to run_diffusion that generates 100 unconditional samples
python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --text_conditional --save_as_json --output_dir "%DIFF_OUTPUT%-unconditional-samples" --game %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "unconditional sampling"

REM calls for evaluating caption adherence
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json %JSON_TEST% --compare_checkpoints --num_tiles %NUM_TILES% --game %GAME%
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json %JSON_TEST% --output_dir "%DIFF_OUTPUT%-caption-adherence-test" --num_tiles %NUM_TILES% --game %GAME%
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json %JSON_RANDOM% --compare_checkpoints --num_tiles %NUM_TILES% --game %GAME%
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json %JSON_RANDOM% --output_dir "%DIFF_OUTPUT%-caption-adherence-random" --num_tiles %NUM_TILES% --game %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "caption adherence evaluation"

REM move the timing log into the trained model's directory
move /Y %TIMING_LOG% "%DIFF_OUTPUT%\pipeline_timing.jsonl"