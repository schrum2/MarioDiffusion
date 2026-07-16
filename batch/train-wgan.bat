@echo off
REM Usage: train-wgan.bat <seed> <data> <game>
REM <seed> is optional, defaults to 0
REM <data> indicates source of data: SMB1, SMB2, etc.
REM <game> Game to train on: Mario, LR, etc
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set DATA=%2

set GAME=%3

set "VALID=false"
for %%G in (Mario LR MM-Simple MM-Full MMLV MM2) do (
    if /I "%GAME%"=="%%~G" set "VALID=true"
)

REM Exit if the flag was never flipped to true
if "%VALID%"=="false" (
    echo Error: Invalid game selected.
    exit /b 1
)

set DIFF_OUTPUT=%GAME%-%DATA%-wgan%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-samples

set DATA_PATH=Game_%GAME%/DATA/%DATA%_LevelsAndCaptions-regular
set TRAIN_DATA=%DATA_PATH%-train.json

python train_wgan.py --augment --game %GAME% --json %TRAIN_DATA% --num_epochs 5000 --nz 32 --output_dir "%DIFF_OUTPUT%" --seed %SEED% --save_image_epochs 10000
python run_wgan.py --game %GAME% --model_path "%DIFF_OUTPUT%\final_models\generator.pth" --num_samples 100 --output_dir "%UNCOND_OUTPUT%" --save_as_json
