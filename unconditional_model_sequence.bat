python create_level_json_data.py --output "SMB1_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros\\Processed"
python create_level_json_data.py --output "SMB2_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros 2 (Japan)\\Processed"
python create_level_json_data.py --output "SML_Levels.json" --levels "..\\TheVGLC\\Super Mario Land\\Processed"
python combine_data.py Mario_Levels.json SMB1_Levels.json SMB2_Levels.json SML_Levels.json
python create_ascii_captions.py --dataset Mario_Levels.json --output Mario_LevelsAndCaptions-regular.json
python tokenizer.py save --json_file Mario_LevelsAndCaptions-regular.json --pkl_file Mario_Tokenizer-regular.pkl
python train_diffusion.py --augment --output_dir "Mario-unconditional-model-regular" --num_epochs 200 --json Mario_LevelsAndCaptions-regular.json --pkl Mario_Tokenizer-regular.pkl 
python run_diffusion.py --model_path Mario-unconditional-model-regular --num_samples 100 --save_as_json --output_dir "Mario-unconditional-model-regular-unconditional-samples"