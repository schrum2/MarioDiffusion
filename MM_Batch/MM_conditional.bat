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
    set RAW_JSON=datasets\MM_Levels_Full.json
    set NUM_TILES=39
) 

else (
    set GAME=MM-Simple
    set DATASET_INFIX=simple
    set TILESET=datasets\MM_Simple_Tileset.json
    set RAW_JSON=datasets\MM_Levels_Simple.json
    set NUM_TILES=13
)

set JSON_TRAIN=datasets\MM_LevelsAndCaptions-%DATASET_INFIX%-regular.json
set JSON_TEST=datasets\MM_LevelsAndCaptions-%DATASET_INFIX%-regular-test.json
set JSON_RANDOM=datasets\MM_RandomTest-%DATASET_INFIX%-regular.json
set PKL=datasets\MM_Tokenizer-%DATASET_INFIX%-regular.pkl
set MLM_OUTPUT=MM-MLM-%DATASET_INFIX%-regular%SEED%
set DIFF_OUTPUT=MM_conditional_%DATASET_INFIX%_regular%SEED%

if "%VARIANT%"=="full" (
    python create_megaman_json_data.py --output %RAW_JSON%
) else (
    python create_megaman_json_data.py --output %RAW_JSON% --group_encodings
)

python MM_create_ascii_captions.py --dataset %RAW_JSON% --tileset %TILESET% --output %JSON_TRAIN%
python tokenizer.py save --json %JSON_TRAIN% --pkl_file %PKL%
python train_mlm.py --epochs 300 --save_checkpoints --json %JSON_TRAIN% --pkl %PKL% --output_dir %MLM_OUTPUT% --seed %SEED%
python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir %DIFF_OUTPUT% --num_epochs 500 --json %JSON_TRAIN% --pkl %PKL% --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% --game %GAME%

REM call to run_diffusion that generates 100 unconditional samples
python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --text_conditional --save_as_json --output_dir "%DIFF_OUTPUT%-unconditional-samples" --game %GAME%

REM calls for evaluating caption adherence
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json %JSON_TEST% --compare_checkpoints --num_tiles %NUM_TILES%
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json %JSON_TEST% --output_dir "%DIFF_OUTPUT%-caption-adherence-test" --num_tiles %NUM_TILES%
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json %JSON_RANDOM% --compare_checkpoints --num_tiles %NUM_TILES%
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json %JSON_RANDOM% --output_dir "%DIFF_OUTPUT%-caption-adherence-random" --num_tiles %NUM_TILES%