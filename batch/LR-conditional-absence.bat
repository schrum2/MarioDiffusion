cd ..

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\\LR_LevelsAndCaptions-absence-train.json --val_json datasets\\LR_LevelsAndCaptions-absence-validate.json --test_json datasets\\LR_LevelsAndCaptions-absence-test.json --pkl datasets\\LR_Tokenizer-absence.pkl --output_dir LR-MLM-absence
python train_diffusion.py --augment --text_conditional --output_dir "LR-conditional-absence" --num_epochs 500 --json datasets\\LR_LevelsAndCaptions-absence-train.json --val_json datasets\\LR_LevelsAndCaptions-absence-validate.json --pkl datasets\\LR_Tokenizer-absence.pkl --mlm_model_dir LR-MLM-absence --plot_validation_caption_score
python run_diffusion.py --model_path LR-conditional-absence --num_samples 100 --text_conditional --save_as_json --output_dir "LR-conditional-absence-unconditional-samples"
python evaluate_caption_adherence.py --model_path LR-conditional-absence --save_as_json --json datasets\\LR_LevelsAndCaptions-absence.json --output_dir text-to-level-final
python evaluate_caption_adherence.py --model_path LR-conditional-absence --save_as_json --json datasets\\LR_LevelsAndCaptions-absence.json --compare_checkpoints
python evaluate_caption_adherence.py --model_path LR-conditional-absence --save_as_json --json datasets\\LR_LevelsAndCaptions-absence-test.json --compare_checkpoints
python evaluate_caption_adherence.py --model_path LR-conditional-absence --save_as_json --json datasets\\LR_RandomTest-absence.json --compare_checkpoints