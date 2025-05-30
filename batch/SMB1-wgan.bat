REM @echo off
REM Usage: SMB1-wgan.bat <seed>
REM <seed> is optional, defaults to 0
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set DIFF_OUTPUT=SMB1-wgan%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-samples

python train_wgan.py --augment --json datasets\\SMB1_LevelsAndCaptions-regular.json --num_epochs 5000 --nz 32 --output_dir "%DIFF_OUTPUT%" --seed %SEED%
python run_wgan.py --model_path "%DIFF_OUTPUT%\final_models\generator.pth" --num_samples 100 --output_dir "%UNCOND_OUTPUT%" --save_as_json
