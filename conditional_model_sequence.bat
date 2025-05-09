python create_level_json_data.py --output "SMB1_Levels.json"
python create_ascii_captions.py --dataset SMB1_Levels.json --output SMB1_LevelsAndCaptions.json
python tokenizer.py save
python train_mlm.py --epochs 300 --save_checkpoints
python create_validation_captions.py --save_file "SMB1_ValidationCaptions.json"
python evaluate_masked_token_prediction.py --model_path mlm --compare_checkpoints
python evaluate_masked_token_prediction.py --model_path mlm --compare_checkpoints --json SMB1_ValidationCaptions.json
python train_diffusion.py --augment --text_conditional --output_dir "conditional-model" --num_epochs 200
python run_diffusion.py --model_path conditional-model --num_samples 100 --text_conditional --save_as_json --output_dir "conditional-model-unconditional-samples"
python evaluate_caption_adherence.py --model_path conditional-model --save_as_json --output_dir conditional-model-text-to-level-final
python evaluate_caption_adherence.py --model_path conditional-model --save_as_json --compare_checkpoints
python evaluate_caption_adherence.py --model_path conditional-model --save_as_json --compare_checkpoints --json SMB1_ValidationCaptions.json