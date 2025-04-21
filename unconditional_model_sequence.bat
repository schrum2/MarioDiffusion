python create_level_json_data.py --output "SMB1_Levels.json"
python create_ascii_captions.py --dataset SMB1_Levels.json --output SMB1_LevelsAndCaptions.json
python tokenizer.py save
python train_diffusion.py --augment --output_dir "unconditional-model" --num_epochs 200
python run_diffusion.py --model_path unconditional-model --num_samples 100 --save_as_json --output_dir "unconditional_model_samples"