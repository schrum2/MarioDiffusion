@echo off


set width=%1
set height=%1
if "%width%"=="" set width=16
if "%height%"=="" set height=14

set width_suffix=%width%
if "%width%"=="" set width_suffix=

cd ..
python create_megaman_json_data.py --output datasets\MM_Levels-simple%width_suffix%.json --target_height %height% --target_width %width% --group_encodings

python MM_create_ascii_captions.py --dataset datasets\MM_Levels-simple%width_suffix%.json --tileset datasets\MM-simple-tileset.json --output datasets\MM_LevelsAndCaptions-simple%width_suffix%-regular.json

python tokenizer.py save --json_file datasets\MM_LevelsAndCaptions-simple%width_suffix%-regular.json --pkl_file datasets\MM_Tokenizer-simple%width_suffix%-regular.pkl

python create_random_test_captions.py --save_file datasets\MM_RandomTest_simple%width_suffix%-regular.json --json datasets\MM_LevelsAndCaptions-simple%width_suffix%-regular.json --seed 0 --game MM-Simple

python split_data.py --json_file datasets\MM_LevelsAndCaptions-simple%width_suffix%-regular.json --train_pct .9 --val_pct .05 --test_pct .05 --seed 0 --game mm-simple

python train_diffusion.py --game MM-Simple --augment --output_dir MM-simple%width_suffix%-unconditional0 --num_epochs 500 --json datasets\MM_LevelsAndCaptions-simple%width_suffix%-regular-train.json --val_json datasets\MM_LevelsAndCaptions-simple%width_suffix%-regular-validate.json --seed 0