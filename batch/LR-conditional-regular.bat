cd ..

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\\LR_LevelsAndCaptions-regular-train.json --val_json datasets\\SMB1_LevelsAndCaptions-regular-validate.json --test_json datasets\\SMB1_LevelsAndCaptions-regular-test.json --pkl datasets\\SMB1_Tokenizer-regular.pkl --output_dir SMB1-MLM-regular
python train_diffusion.py --augment --text_conditional --output_dir "LR-conditional-regular" --num_epochs 500 --json datasets\\LR_LevelsAndCaptions-regular-train.json --val_json datasets\\LR_LevelsAndCaptions-regular-validate.json --pkl datasets\\LR_Tokenizer-regular.pkl --mlm_model_dir SMB1-MLM-regular --plot_validation_caption_score
python run_diffusion.py --model_path LR-conditional-regular --num_samples 100 --text_conditional --save_as_json --output_dir "LR-conditional-regular-unconditional-samples"
python evaluate_caption_adherence.py --model_path LR-conditional-regular --save_as_json --json datasets\\LR_LevelsAndCaptions-regular.json --output_dir text-to-level-final --num_tiles 8
python evaluate_caption_adherence.py --model_path LR-conditional-regular --save_as_json --json datasets\\LR_LevelsAndCaptions-regular.json --compare_checkpoints --num_tiles 8
python evaluate_caption_adherence.py --model_path LR-conditional-regular --save_as_json --json datasets\\LR_LevelsAndCaptions-regular-test.json --compare_checkpoints --num_tiles 8
python evaluate_caption_adherence.py --model_path LR-conditional-regular --save_as_json --json datasets\\LR_RandomTest-regular.json --compare_checkpoints --num_tiles 8