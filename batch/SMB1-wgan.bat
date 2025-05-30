REM @echo off
REM Usage: SMB1-wgan.bat <type> <seed>
REM <type> should be "regular" 
REM <seed> is optional, defaults to 0
cd ..

set TYPE=%1
if "%TYPE%"=="" set TYPE=regular

set SEED=%2
if "%SEED%"=="" set SEED=0

set DIFF_OUTPUT=SMB1-wgan%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-samples

python train_wgan.py --augment --json datasets\\SMB1_LevelsAndCaptions-regular.json --num_epochs 1000 --nz 32 --output_dir "%DIFF_OUTPUT%" --seed %SEED%
python run_wgan.py --model_path "%DIFF_OUTPUT%\final_models\generator.pth" --num_samples 100 --output_dir "%UNCOND_OUTPUT%" --save_as_json
