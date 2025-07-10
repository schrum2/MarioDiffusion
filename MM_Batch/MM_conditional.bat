
REM @echo off
REM Usage: MM-conditional.bat <seed> <type>
REM <type> should be "regular" or "absence"
REM <seed> is optional, defaults to 0
cd ..

python create_megaman_json_data.py --output datasets\\MM_Levels_Simple.json --group_encodings
python MM_create_ascii_captions.py --dataset datasets\\MM_Levels_Simple.json --tileset datasets\\MM_Simple_Tileset.json --output datasets\\MM_LevelsAndCaptions-simple-regular.json
python tokenizer.py save --json datasets\\MM_LevelsAndCaptions-simple-regular.json --pkl_file datasets\MM_Tokenizer-simple-regular.pkl
python train_mlm.py --json datasets\\MM_LevelsAndCaptions-simple-regular.json --output_dir MM-MLM-simple-regular --save_checkpoints --pkl datasets\MM_Tokenizer-simple-regular.pkl
python train_diffusion.py --pkl datasets\MM_Tokenizer-simple-regular.pkl --json datasets\\MM_LevelsAndCaptions-simple-regular.json --augment --mlm_model_dir MM-MLM-simple-regular --text_conditional --output_dir MM_conditional_simple_regular0 --seed 0 --game MM-Simple
