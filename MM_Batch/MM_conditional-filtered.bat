@echo off

cd ..

set SIZE=%1
if "%SIZE%"=="" (
	set WIDTH=16
	set HEIGHT=14
) else (
	set WIDTH=%SIZE%
	set HEIGHT=%SIZE%
)

python create_megaman_json_data.py --output datasets\MM_Levels-simple%WIDTH%-filtered.json --group_encodings --traversable_only --target_width %WIDTH% --target_height %HEIGHT%

python MM_create_ascii_captions.py --dataset datasets\MM_Levels-simple%WIDTH%-filtered.json --tileset datasets\MM-simple-tileset.json --output datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular.json

python tokenizer.py save --json_file datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular.json --pkl_file datasets\MM_Tokenizer-simple%WIDTH%-filtered-regular.pkl

python create_random_test_captions.py --save_file datasets\MM_RandomTest_simple%WIDTH%-filtered-regular.json --json datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular.json --seed 0 --game MM-Simple

python split_data.py --json_file datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular.json --train_pct .9 --val_pct .05 --test_pct .05 --seed 0 --game mm-simple

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular.json --pkl datasets\MM_Tokenizer-simple%WIDTH%-filtered-regular.pkl --output_dir MM-MLM-simple%WIDTH%-filtered-regular --seed 0

python train_diffusion.py --text_conditional --mlm_model_dir MM-MLM-simple%WIDTH%-filtered-regular --game MM-Simple --augment --output_dir MM-simple%WIDTH%-filtered-conditional --num_epochs 500 --json datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular-train.json --val_json datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular-validate.json --seed 0
