python split_data.py --json SMB1_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42

python train_diffusion.py --augment --text_conditional --output_dir "SMB1-conditional-regular-gte-large" --num_epochs 100 --json SMB1_LevelsAndCaptions-regular-train.json --val_json SMB1_LevelsAndCaptions-regular-validate.json --pkl SMB1_Tokenizer-regular.pkl --pretrained_language_model Alibaba-NLP/gte-large-en-v1.5 --plot_validation_caption_score
python run_diffusion.py --model_path SMB1-conditional-regular-gte-large --num_samples 100 --text_conditional --save_as_json --output_dir "SMB1-conditional-regular-gte-large-unconditional-samples"
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-gte-large --save_as_json --json SMB1_LevelsAndCaptions-regular.json --output_dir text-to-level-final
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-gte-large --save_as_json --json SMB1_LevelsAndCaptions-regular.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-gte-large --save_as_json --json SMB1_ValidationCaptions-regular.json --compare_checkpoints 
