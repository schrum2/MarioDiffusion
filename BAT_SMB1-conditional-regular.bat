python train_mlm.py --epochs 300 --save_checkpoints --json SMB1_LevelsAndCaptions-regular.json --pkl SMB1_Tokenizer-regular.pkl --output_dir SMB1-MLM-regular --split
python split_data.py --json SMB1_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python train_diffusion.py --augment --text_conditional --output_dir "SMB1-conditional-regular" --num_epochs 100 --json SMB1_LevelsAndCaptions-regular-train.json --val_json SMB1_LevelsAndCaptions-regular-validate.json --pkl SMB1_Tokenizer-regular.pkl --mlm_model_dir SMB1-MLM-regular --plot_validation_caption_score
python run_diffusion.py --model_path SMB1-conditional-regular --num_samples 100 --text_conditional --save_as_json --output_dir "SMB1-conditional-regular-unconditional-samples"
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular --save_as_json --json SMB1_LevelsAndCaptions-regular.json --output_dir text-to-level-final
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular --save_as_json --json SMB1_LevelsAndCaptions-regular.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular --save_as_json --json SMB1_ValidationCaptions-regular.json --compare_checkpoints 
