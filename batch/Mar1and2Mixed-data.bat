


call Mar1and2-data.bat 32
cd batch
call Mar1and2-data.bat 64
cd batch
call Mar1and2-data.bat 128


python combine_data.py datasets\\Mar1and2_16-32_LevelsAndCaptions-regular.json datasets\\Mar1and2_LevelsAndCaptions-regular.json datasets\\Mar1and2_32_LevelsAndCaptions-regular.json
python combine_data.py datasets\\Mar1and2_64-128_LevelsAndCaptions-regular.json datasets\\Mar1and2_64_LevelsAndCaptions-regular.json datasets\\Mar1and2_128_LevelsAndCaptions-regular.json
python combine_data.py datasets\\Mar1and2_16-32-64-128_LevelsAndCaptions-regular.json datasets\\Mar1and2_16-32_LevelsAndCaptions-regular.json datasets\\Mar1and2_64-128_LevelsAndCaptions-regular.json
python split_data.py --json_file datasets\\Mar1and2_16-32-64-128_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game mario

python tokenizer.py save --json_file datasets\Mar1and2_16-32-64-128_LevelsAndCaptions-regular.json --pkl_file datasets\Mar1and2Mixed_Tokenizer-regular.pkl
python train_mlm.py --epochs 300 --save_checkpoints --json  datasets\Mar1and2_16-32-64-128_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_16-32-64-128_LevelsAndCaptions-regular-validate.json --test_json datasets\Mar1and2_16-32-64-128_LevelsAndCaptions-regular-test.json --pkl datasets\Mar1and2Mixed_Tokenizer-regular.pkl --output_dir Mar1and2Mixed-MLM-regular0 --seed 0
python train_diffusion.py --augment --text_conditional --output_dir Mar1and2_mixed-conditional0 --num_epochs 200 --json datasets\\Mar1and2_16-32-64-128_LevelsAndCaptions-regular-train.json --val_json datasets\\Mar1and2_16-32-64-128_LevelsAndCaptions-regular-validate.json --pkl datasets\Mar1and2Mixed_Tokenizer-regular.pkl --mlm_model_dir Mar1and2Mixed-MLM-regular0 --plot_validation_caption_score --seed 0 --batch_size 16 --save_image_epochs 5