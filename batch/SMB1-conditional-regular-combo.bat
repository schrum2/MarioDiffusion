cd ..

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\\SMB1_LevelsAndCaptions-regular-train.json --val_json datasets\\SMB1_LevelsAndCaptions-regular-validate.json --test_json datasets\\SMB1_LevelsAndCaptions-regular-test.json --pkl datasets\\SMB1_Tokenizer-regular.pkl --output_dir SMB1-MLM-regular
python train_diffusion.py --augment --text_conditional --output_dir "SMB1-conditional-regular-combo" --num_epochs 500 --json datasets\\SMB1_LevelsAndCaptions-regular-train.json --val_json datasets\\SMB1_LevelsAndCaptions-regular-validate.json --pkl datasets\\SMB1_Tokenizer-regular.pkl --mlm_model_dir SMB1-MLM-regular --plot_validation_caption_score --loss_type COMBO
python run_diffusion.py --model_path SMB1-conditional-regular-combo --num_samples 100 --text_conditional --save_as_json --output_dir "SMB1-conditional-regular-unconditional-combo-samples"
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-combo --save_as_json --json datasets\\SMB1_LevelsAndCaptions-regular.json --output_dir text-to-level-final
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-combo --save_as_json --json datasets\\SMB1_LevelsAndCaptions-regular.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-combo --save_as_json --json datasets\\SMB1_ValidationCaptions-regular.json --compare_checkpoints 
