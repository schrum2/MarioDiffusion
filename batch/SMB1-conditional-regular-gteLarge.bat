cd ..

python train_diffusion.py --augment --text_conditional --output_dir "SMB1-conditional-regular-gteLarge" --num_epochs 500 --json datasets\\SMB1_LevelsAndCaptions-regular-train.json --val_json datasets\\SMB1_LevelsAndCaptions-regular-validate.json --pretrained_language_model "Alibaba-NLP/gte-large-en-v1.5" --plot_validation_caption_score
python run_diffusion.py --model_path SMB1-conditional-regular-gteLarge --num_samples 100 --text_conditional --save_as_json --output_dir "SMB1-conditional-regular-gteLarge-unconditional-samples"
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-gteLarge --save_as_json --json datasets\\SMB1_LevelsAndCaptions-regular.json --output_dir text-to-level-final
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-gteLarge --save_as_json --json datasets\\SMB1_LevelsAndCaptions-regular.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path SMB1-conditional-regular-gteLarge --save_as_json --json datasets\\SMB1_ValidationCaptions-regular.json --compare_checkpoints
