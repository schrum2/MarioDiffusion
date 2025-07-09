REM @echo off
REM Usage: train-wgan.bat <seed>
REM <seed> is optional, defaults to 0
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set GAME=LR

set DIFF_OUTPUT=%GAME%-wgan%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-samples

python train_wgan.py --augment --json datasets\%GAME%_LevelsAndCaptions-regular.json --num_epochs 20000 --nz 10 --output_dir "%DIFF_OUTPUT%" --seed %SEED% --save_image_epochs 20 --game %GAME%
python run_wgan.py --model_path "%DIFF_OUTPUT%\final_models\generator.pth" --num_samples 100 --output_dir "%UNCOND_OUTPUT%" --save_as_json --game %GAME% --nz 10
