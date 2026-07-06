@echo off

cd ..

python megaman\bulk_mmlv_to_vglc.py --output datasets\MMLV_Maker_Levels

python create_megaman_json_data.py --levels datasets\MMLV_Maker_Levels --tileset datasets\MMLV.json --stride_x 16 --stride_y 14 --scan_mode snap --max_enemies 4 --min_content_pct 15 --direction_captions --include_moving_ground --output datasets\MMLV_Levels.json

python MM_create_ascii_captions.py --dataset datasets\MMLV_Levels.json --tileset datasets\MMLV.json --output datasets\MMLV_LevelsAndCaptions-regular.json

python tokenizer.py save --json datasets\MMLV_LevelsAndCaptions-regular.json --pkl_file datasets\MMLV_Tokenizer-regular.pkl

python create_random_test_captions.py --save_file datasets\MMLV_RandomTest-regular.json --json datasets\MMLV_LevelsAndCaptions-regular.json --seed 0 --game MMLV

python split_data.py --json_file datasets\MMLV_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MMLV