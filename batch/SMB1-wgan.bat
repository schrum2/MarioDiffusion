cd ..


python train_wgan.py --augment --json datasets\\SMB1_LevelsAndCaptions-regular.json --num_epochs 1000 --nz 32 --output_dir "SMB1-WGAN-regular"
python run_wgan.py --model_path "SMB1-WGAN-regular\final_models\generator.pth" --num_samples 100 --pkl datasets\\SMB1_Tokenizer-regular.pkl --output_dir "SMB1-WGAN-Samples" --save_as_json