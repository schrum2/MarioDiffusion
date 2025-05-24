python train_mlm.py --epochs 300 --save_checkpoints --json SMB1_LevelsAndCaptions-absence.json --pkl SMB1_Tokenizer-absence.pkl --output_dir SMB1-MLM-absence --split
python split_data.py --json SMB1_LevelsAndCaptions-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python train_diffusion.py --augment --text_conditional --output_dir "SMB1-conditional-absence" --num_epochs 100 --json SMB1_LevelsAndCaptions-absence-train.json --val_json SMB1_LevelsAndCaptions-absence-validate.json --pkl SMB1_Tokenizer-absence.pkl --mlm_model_dir SMB1-MLM-absence --plot_validation_caption_score
python run_diffusion.py --model_path SMB1-conditional-absence --num_samples 100 --text_conditional --save_as_json --output_dir "SMB1-conditional-absence-unconditional-samples"
python evaluate_caption_adherence.py --model_path SMB1-conditional-absence --save_as_json --json SMB1_LevelsAndCaptions-absence.json --output_dir text-to-level-final
python evaluate_caption_adherence.py --model_path SMB1-conditional-absence --save_as_json --json SMB1_LevelsAndCaptions-absence.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path SMB1-conditional-absence --save_as_json --json SMB1_ValidationCaptions-absence.json --compare_checkpoints 
