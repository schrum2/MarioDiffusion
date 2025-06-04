REM @echo off
REM Usage: train-wgan.bat <seed> <game>
REM <seed> is optional, defaults to 0
REM <game> indicates source of data: SMB1, SMB2, etc.
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set GAME=%2

set DIFF_OUTPUT=%GAME%-wgan%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-samples

python train_wgan.py --augment --json datasets\%GAME%_LevelsAndCaptions-regular.json --num_epochs 5000 --nz 32 --output_dir "%DIFF_OUTPUT%" --seed %SEED%
python run_wgan.py --model_path "%DIFF_OUTPUT%\final_models\generator.pth" --num_samples 100 --output_dir "%UNCOND_OUTPUT%" --save_as_json
