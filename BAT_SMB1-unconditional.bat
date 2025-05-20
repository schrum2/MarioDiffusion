python train_diffusion.py --augment --output_dir "SMB1-unconditional" --num_epochs 100 --json SMB1_LevelsAndCaptions-regular.json --split
python run_diffusion.py --model_path SMB1-unconditional --num_samples 100 --save_as_json --output_dir "SMB1-unconditional-samples"
