cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

python -m mm2pipeline_data extract --output_folder MM2_Data\bcd --limit 25000 --likes 10 --skip_3dworld --skip_items --skip_subworld_items

python -m mm2pipeline_data toost --input MM2_Data\bcd -o MM2_Data\json --images-output MM2_Data\images


python -m mm2pipeline_data json-to-ascii --input MM2_Data\json --output_folder MM2_Data\ascii


python -m mm2pipeline_data dataset build --input MM2_Data\ascii --output_folder datasets\MM2_Levels-regular.json --tileset datasets\mm2_tileset_we.json --sliding_window --stride 20 --strip_goal

python MM2_Files\MarioMaker_create_ascii_captions.py --dataset datasets\MM2_Levels-regular.json --tileset datasets\mm2_tileset_we.json --output datasets\MM2_LevelsAndCaptions-regular.json

python tokenizer.py save --json_file datasets\MM2_LevelsAndCaptions-regular.json --pkl_file datasets\MM2_Tokenizer-regular.pkl

python create_random_test_captions.py --save_file datasets\MM2_RandomTest-regular.json --json datasets\MM2_LevelsAndCaptions-regular.json --seed %SEED% --game MM2

python split_data.py --json_file datasets\MM2_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MM2
