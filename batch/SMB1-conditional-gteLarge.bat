REM @echo off
REM Usage: SMB1-conditional-gteLarge.bat <seed> <type>
REM <seed> is optional, defaults to 0
REM <type> should be "regular" or "absence"
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set TYPE=%2
if "%TYPE%"=="" set TYPE=regular

set DIFF_OUTPUT=SMB1-conditional-%TYPE%-gteLarge%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-unconditional-samples

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

python train_diffusion.py --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\SMB1_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\SMB1_LevelsAndCaptions-%TYPE%-validate.json --pretrained_language_model "Alibaba-NLP/gte-large-en-v1.5" --plot_validation_caption_score --seed %SEED%
python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --text_conditional --save_as_json --output_dir "%UNCOND_OUTPUT%" %DESCRIBE_ABSENCE_FLAG%
call batch\evaluate_caption_adherence_multi.bat %DIFF_OUTPUT% %TYPE%