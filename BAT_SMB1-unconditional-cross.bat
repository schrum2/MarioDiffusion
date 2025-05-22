python train_diffusion.py --augment --output_dir "SMB1-unconditional-cross" --num_epochs 500 --json SMB1_LevelsAndCaptions-regular.json --split --loss_type CROSS
python run_diffusion.py --model_path SMB1-unconditional-cross --num_samples 100 --save_as_json --output_dir "SMB1-unconditional-cross-samples"
