python train_diffusion.py --augment --output_dir "SMB1-unconditional-combo" --num_epochs 500 --json SMB1_LevelsAndCaptions-regular.json --split --loss_type COMBO
IF %ERRORLEVEL% NEQ 0 EXIT /B %ERRORLEVEL%
python run_diffusion.py --model_path SMB1-unconditional-combo --num_samples 100 --save_as_json --output_dir "SMB1-unconditional-combo-samples"
