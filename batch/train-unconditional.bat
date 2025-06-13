REM @echo off
REM Usage: train-ununconditional.bat <seed> <game>
REM <seed> is optional, defaults to 0
REM <game> indicates source of data: SMB1, SMB2, etc.
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set GAME=%2

set GAME_PLAYED=
if /I "%GAME%"=="LR" set GAME_PLAYED=--game LR

set DIFF_OUTPUT=%GAME%-unconditional%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-samples

python train_diffusion.py --augment --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\%GAME%_LevelsAndCaptions-regular-train.json --val_json datasets\%GAME%_LevelsAndCaptions-regular-validate.json --seed %SEED% %GAME_PLAYED%
call batch\run_diffusion_multi.bat %DIFF_OUTPUT% %TYPE% %GAME%
