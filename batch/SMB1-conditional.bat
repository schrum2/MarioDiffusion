REM @echo off
REM Usage: SMB1-conditional.bat <type> <seed>
REM <type> should be "regular", "absence", or "negative"
REM <seed> is optional, defaults to 0
cd ..

set TYPE=%1
if "%TYPE%"=="" set TYPE=regular

set SEED=%2
if "%SEED%"=="" set SEED=0

REM Set up variables for all cases
set MLM_OUTPUT=SMB1-MLM-%TYPE%%SEED%

REM Default values for conditional model output and extra flags
set DIFF_OUTPUT=SMB1-conditional-%TYPE%%SEED%
set DIFF_FLAGS=
set UNCOND_OUTPUT=%DIFF_OUTPUT%-unconditional-samples

REM Special case for negative prompt training
if /I "%TYPE%"=="negative" (
    set TYPE=regular
    set DIFF_FLAGS=--negative_prompt_training
)

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\SMB1_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\SMB1_LevelsAndCaptions-%TYPE%-validate.json --test_json datasets\SMB1_LevelsAndCaptions-%TYPE%-test.json --pkl datasets\SMB1_Tokenizer-%TYPE%.pkl --output_dir %MLM_OUTPUT% --seed %SEED%
python train_diffusion.py --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\SMB1_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\SMB1_LevelsAndCaptions-%TYPE%-validate.json --pkl datasets\SMB1_Tokenizer-%TYPE%.pkl --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% %DIFF_FLAGS%
python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --text_conditional --save_as_json --output_dir "%UNCOND_OUTPUT%"
call batch\\evaluate_caption_adherence_multi.bat %DIFF_OUTPUT% %TYPE%