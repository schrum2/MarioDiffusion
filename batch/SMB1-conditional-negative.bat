@echo off
REM Usage: SMB1-conditional-negative.bat <type> <seed>
REM <type> should be "regular" or "absence" (should simply be regular)
REM <seed> is optional, defaults to 0
cd ..

set TYPE=%1
if "%TYPE%"=="" set TYPE=regular

set SEED=%2
if "%SEED%"=="" set SEED=0

set MLM_OUTPUT=SMB1-MLM-%TYPE%%SEED%
set DIFF_OUTPUT=SMB1-conditional-negative%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-unconditional-samples

python train_diffusion.py --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\SMB1_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\SMB1_LevelsAndCaptions-%TYPE%-validate.json --pkl datasets\SMB1_Tokenizer-%TYPE%.pkl --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% --negative_prompt_training
python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --text_conditional --save_as_json --output_dir "%UNCOND_OUTPUT%"
call evaluate_caption_adherence_multi.bat %DIFF_OUTPUT% %TYPE%