cd ..


python run_gpt2.py --output_dir "MarioGPT_Levels"
python create_level_json_data.py --output "datasets\\MarioGPT_Levels.json" --levels "MarioGPT_Levels\levels" --stride 16
python create_ascii_captions.py --dataset "datasets\\MarioGPT_Levels.json" --output "datasets\\MarioGPT_LevelsAndCaptions-regular.json"
python calculate_gpt2_metrics.py --generated_levels "datasets\\MarioGPT_LevelsAndCaptions-regular.json" --training_levels "datasets\\Mar1and2_LevelsAndCaptions-regular.json" --full_levels_dir "MarioGPT_Levels//levels" --output_dir "MarioGPT_metrics"