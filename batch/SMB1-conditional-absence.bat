cd ..

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\\SMB1_LevelsAndCaptions-absence-train.json --val_json datasets\\SMB1_LevelsAndCaptions-absence-validate.json --test_json datasets\\SMB1_LevelsAndCaptions-absence-test.json --pkl datasets\\SMB1_Tokenizer-absence.pkl --output_dir SMB1-MLM-absence
python train_diffusion.py --augment --text_conditional --output_dir "SMB1-conditional-absence" --num_epochs 500 --json datasets\\SMB1_LevelsAndCaptions-absence-train.json --val_json datasets\\SMB1_LevelsAndCaptions-absence-validate.json --pkl datasets\\SMB1_Tokenizer-absence.pkl --mlm_model_dir SMB1-MLM-absence --plot_validation_caption_score
python run_diffusion.py --model_path SMB1-conditional-absence --num_samples 100 --text_conditional --save_as_json --output_dir "SMB1-conditional-absence-unconditional-samples"
python evaluate_caption_adherence.py --model_path SMB1-conditional-absence --save_as_json --json datasets\\SMB1_LevelsAndCaptions-absence.json --output_dir text-to-level-final
python evaluate_caption_adherence.py --model_path SMB1-conditional-absence --save_as_json --json datasets\\SMB1_LevelsAndCaptions-absence.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path SMB1-conditional-absence --save_as_json --json datasets\\SMB1_ValidationCaptions-absence.json --compare_checkpoints 
