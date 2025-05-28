cd ..

python train_diffusion.py --augment --text_conditional --output_dir "SMB1-conditional-regular-MiniLM-split" --num_epochs 500 --json datasets\\SMB1_LevelsAndCaptions-regular-train.json --val_json datasets\\SMB1_LevelsAndCaptions-regular-validate.json --pretrained_language_model "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" --split_pretrained_sentences --plot_validation_caption_score
python run_diffusion.py --model_path SMB1-conditional-regular-MiniLM --num_samples 100 --text_conditional --save_as_json --output_dir "SMB1-conditional-regular-MiniLM-unconditional-samples"
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-MiniLM --save_as_json --json datasets\\SMB1_LevelsAndCaptions-regular.json --output_dir text-to-level-final
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-MiniLM --save_as_json --json datasets\\SMB1_LevelsAndCaptions-regular.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-MiniLM --save_as_json --json datasets\\SMB1_LevelsAndCaptions-regular-test.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-MiniLM --save_as_json --json datasets\\SMB1_RandomTest-regular.json --compare_checkpoints
