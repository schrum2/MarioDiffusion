python create_level_json_data.py --output "LR_Levels.json" --levels "..\\TheVGLC\\Lode Runner\\Processed" --tileset "..\\TheVGLC\\Lode Runner\\Loderunner.json" --target_height 32 --target_width 32 --extra_tile .
python LR_create_ascii_captions.py --dataset LR_Levels.json --output LR_LevelsAndCaptions-regular.json
python train_diffusion.py --augment --output_dir "LR-unconditional-regular" --num_epochs 100 --json LR_LevelsAndCaptions-regular.json --split --num_tiles 10 --batch_size 5 --game LR
python run_diffusion.py --model_path LR-unconditional-regular --num_samples 100 --save_as_json --output_dir "LR-unconditional-regular-samples" --game LR
