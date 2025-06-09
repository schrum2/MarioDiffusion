REM @echo off
REM Usage: train-conditional.bat <seed> <game> <type> 
REM <seed> is optional, defaults to 0
REM <game> indicates source of data: SMB1, SMB2, etc.
REM <type> should be "regular", "absence", or "negative"
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set GAME=%2

set TYPE=%3
if "%TYPE%"=="" set TYPE=regular

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

REM Set up variables for all cases
set MLM_OUTPUT=%GAME%-MLM-%TYPE%%SEED%

REM Default values for conditional model output and extra flags
set DIFF_OUTPUT=%GAME%-conditional-%TYPE%%SEED%
set DIFF_FLAGS=

REM Special case for negative prompt training
if /I "%TYPE%"=="negative" (
    set TYPE=regular
    set DIFF_FLAGS=--negative_prompt_training
)

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --test_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-test.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --output_dir %MLM_OUTPUT% --seed %SEED%
python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% %DIFF_FLAGS% %DESCRIBE_ABSENCE_FLAG%
call batch\run_diffusion_multi.bat %DIFF_OUTPUT% %TYPE% text
call batch\evaluate_caption_adherence_multi.bat %DIFF_OUTPUT% %TYPE% %GAME%
