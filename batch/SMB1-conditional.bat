REM @echo off
REM Usage: SMB1-conditional.bat <seed> <type> 
REM <seed> is optional, defaults to 0
REM <type> should be "regular", "absence", or "negative"
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set TYPE=%2
if "%TYPE%"=="" set TYPE=regular

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

REM Set up variables for all cases
set MLM_OUTPUT=SMB1-MLM-%TYPE%%SEED%

REM Default values for conditional model output and extra flags
set DIFF_OUTPUT=SMB1-conditional-%TYPE%%SEED%
set DIFF_FLAGS=

REM Special case for negative prompt training
if /I "%TYPE%"=="negative" (
    set TYPE=regular
    set DIFF_FLAGS=--negative_prompt_training
)

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\SMB1_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\SMB1_LevelsAndCaptions-%TYPE%-validate.json --test_json datasets\SMB1_LevelsAndCaptions-%TYPE%-test.json --pkl datasets\SMB1_Tokenizer-%TYPE%.pkl --output_dir %MLM_OUTPUT% --seed %SEED%
python train_diffusion.py --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\SMB1_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\SMB1_LevelsAndCaptions-%TYPE%-validate.json --pkl datasets\SMB1_Tokenizer-%TYPE%.pkl --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% %DIFF_FLAGS% %DESCRIBE_ABSENCE_FLAG%
python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --text_conditional --save_as_json --output_dir "%UNCOND_OUTPUT%" %DESCRIBE_ABSENCE_FLAG%
call batch\run_diffusion_multi.bat %DIFF_OUTPUT% %TYPE% text
call batch\evaluate_caption_adherence_multi.bat %DIFF_OUTPUT% %TYPE%