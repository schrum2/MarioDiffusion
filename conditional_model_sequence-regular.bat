python create_level_json_data.py --output "SMB1_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros\\Processed"
python create_level_json_data.py --output "SMB2_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros 2 (Japan)\\Processed"
python create_level_json_data.py --output "SML_Levels.json" --levels "..\\TheVGLC\\Super Mario Land\\Processed"
python combine_data.py Mario_Levels.json SMB1_Levels.json SMB2_Levels.json SML_Levels.json
python create_ascii_captions.py --dataset Mario_Levels.json --output Mario_LevelsAndCaptions-regular.json
python tokenizer.py save --json_file Mario_LevelsAndCaptions-regular.json --pkl_file Mario_Tokenizer-regular.pkl
python train_mlm.py --epochs 300 --save_checkpoints --json Mario_LevelsAndCaptions-regular.json --pkl Mario_Tokenizer-regular.pkl --output_dir mlm-regular
python create_validation_captions.py --save_file "Mario_ValidationCaptions-regular.json" --pkl Mario_Tokenizer-regular.pkl --json Mario_LevelsAndCaptions-regular.json  --seed 0
python evaluate_masked_token_prediction.py --model_path mlm-regular --compare_checkpoints --json Mario_LevelsAndCaptions-regular.json
python evaluate_masked_token_prediction.py --model_path mlm-regular --compare_checkpoints --json Mario_ValidationCaptions-regular.json
python train_diffusion.py --augment --text_conditional --output_dir "Mario-conditional-model-regular" --num_epochs 200 --json Mario_LevelsAndCaptions-regular.json --pkl Mario_Tokenizer-regular.pkl --mlm_model_dir mlm-regular 
python run_diffusion.py --model_path Mario-conditional-model-regular --num_samples 100 --text_conditional --save_as_json --output_dir "Mario-conditional-model-regular-unconditional-samples"
python evaluate_caption_adherence.py --model_path Mario-conditional-model-regular --save_as_json --json Mario_LevelsAndCaptions-regular.json --output_dir conditional-model-text-to-level-final
python evaluate_caption_adherence.py --model_path Mario-conditional-model-regular --save_as_json --json Mario_LevelsAndCaptions-regular.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path Mario-conditional-model-regular --save_as_json --json Mario_ValidationCaptions-regular.json --compare_checkpoints 
