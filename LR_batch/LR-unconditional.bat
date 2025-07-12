REM @echo off
REM Usage: LR-ununconditional.bat <seed> <type>
REM <seed> is optional, defaults to 0
REM <type> should be "regular" or "absence"
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set TYPE=%2
if "%TYPE%"=="" set TYPE=regular


set DIFF_OUTPUT=LR-unconditional%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-samples

python train_diffusion.py --augment --output_dir "%DIFF_OUTPUT%" --num_epochs 3000 --json datasets\LR_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\LR_LevelsAndCaptions-%TYPE%-validate.json --seed %SEED% --game LR
python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --save_as_json --output_dir "%UNCOND_OUTPUT%" --game LR
