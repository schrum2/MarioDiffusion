cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

python -m mm2pipeline_data extract --output_folder MM2_Data\bcd --limit 25000 --likes 10 --skip_3dworld --skip_items --skip_subworld_items

python -m mm2pipeline_data toost --input MM2_Data\bcd -o MM2_Data\json --images-output MM2_Data\images


python -m mm2pipeline_data json-to-ascii --input MM2_Data\json --output_folder MM2_Data\ascii


python -m mm2pipeline_data dataset build --input MM2_Data\ascii --output_folder datasets\MM2_Levels-regular.json --tileset Game_MM2\mm2_tileset_we.json --sliding_window --stride 20 --strip_goal

python MM2_Files\MarioMaker_create_ascii_captions.py --dataset datasets\MM2_Levels-regular.json --tileset Game_MM2\mm2_tileset_we.json --output datasets\MM2_LevelsAndCaptions-regular.json

python tokenizer.py save --json_file datasets\MM2_LevelsAndCaptions-regular.json --pkl_file datasets\MM2_Tokenizer-regular.pkl

python create_random_test_captions.py --save_file datasets\MM2_RandomTest-regular.json --json datasets\MM2_LevelsAndCaptions-regular.json --seed %SEED% --game MM2

python split_data.py --json_file datasets\MM2_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MM2

set GAME=MM2
set NUM_TILES=67
set JSON_TRAIN=datasets\MM2_LevelsAndCaptions-regular-train.json
set JSON_VAL=datasets\MM2_LevelsAndCaptions-regular-validate.json
set JSON_TEST=datasets\MM2_LevelsAndCaptions-regular-test.json
set JSON_RANDOM=datasets\MM2_RandomTest-regular.json
set PKL=datasets\MM2_Tokenizer-regular.pkl
set MLM_OUTPUT=MM2-MLM-regular%SEED%
set DIFF_OUTPUT=MM2-conditional-regular%SEED%

python train_mlm.py --epochs 300 --save_checkpoints --json %JSON_TRAIN% --val_json %JSON_VAL% --test_json %JSON_TEST% --pkl %PKL% --output_dir %MLM_OUTPUT% --seed %SEED%

python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir %DIFF_OUTPUT% --num_epochs 500 --json %JSON_TRAIN% --val_json %JSON_VAL% --pkl %PKL% --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% --game %GAME%

python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --text_conditional --save_as_json --output_dir "%DIFF_OUTPUT%-samples" --game %GAME%

python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json %JSON_TEST% --compare_checkpoints --num_tiles %NUM_TILES% --game %GAME%
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json %JSON_RANDOM% --output_dir "%DIFF_OUTPUT%-caption-adherence-random" --num_tiles %NUM_TILES% --game %GAME%

python -m MM2_Files.evaluate_mm2_metrics --model_path %DIFF_OUTPUT% --game %GAME%
