@echo off
cd ..

REM Get the first argument as SEED, default to 0 if not provided
set SEED=%1
if "%SEED%"=="" set SEED=0

REM Append SEED to output_dir names
set MLM_OUTPUT=SMB1-MLM-regular%SEED%
set DIFF_OUTPUT=SMB1-conditional-regular%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-unconditional-samples

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\SMB1_LevelsAndCaptions-regular-train.json --val_json datasets\SMB1_LevelsAndCaptions-regular-validate.json --test_json datasets\SMB1_LevelsAndCaptions-regular-test.json --pkl datasets\SMB1_Tokenizer-regular.pkl --output_dir %MLM_OUTPUT% --seed %SEED%
python train_diffusion.py --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\SMB1_LevelsAndCaptions-regular-train.json --val_json datasets\SMB1_LevelsAndCaptions-regular-validate.json --pkl datasets\SMB1_Tokenizer-regular.pkl --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED%
python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --text_conditional --save_as_json --output_dir "%UNCOND_OUTPUT%"
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json datasets\SMB1_LevelsAndCaptions-regular.json --output_dir text-to-level-final
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json datasets\SMB1_LevelsAndCaptions-regular.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json datasets\SMB1_LevelsAndCaptions-regular-test.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json datasets\SMB1_RandomTest-regular.json --compare_checkpoints