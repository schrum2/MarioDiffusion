python create_level_json_data.py --output "SMB1_Levels.json"
python create_ascii_captions.py --dataset SMB1_Levels.json --output SMB1_LevelsAndCaptions.json
python validation_captions.py --save_file "SMB1_ValidationCaptions.json"
python tokenizer.py save
python train_mlm.py --epochs 300
python train_diffusion.py --augment --text_conditional --output_dir "conditional-model" --num_epochs 200
python run_diffusion.py --model_path conditional-model --num_samples 100 --text_conditional --save_as_json --output_dir "conditional_model_unconditional_samples"
python evaluate_caption_adherence.py --model_path conditional-model --save_as_json
python evaluate_caption_adherence.py --model_path conditional-model --save_as_json --compare_checkpoints --output_dir text_to_level_training
python evaluate_caption_adherence.py --model_path conditional-model --save_as_json --compare_checkpoints --json SMB1_ValidationCaptions.json --output_dir text_to_level_validation