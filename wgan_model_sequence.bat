python create_level_json_data.py --output "SMB1_Levels.json"
python create_ascii_captions.py --dataset SMB1_Levels.json --output SMB1_LevelsAndCaptions.json
python tokenizer.py save
python train_wgan.py --augment
python run_wgan.py --model_path wgan-output\final_models\generator.pth --num_samples 100 --output_dir wgan_samples