REM @echo off
REM Usage: SMB1-ununconditional.bat <seed> 
REM <seed> is optional, defaults to 0
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set DIFF_OUTPUT=SMB1-unconditional%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-samples

python train_diffusion.py --augment --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\SMB1_LevelsAndCaptions-regular-train.json --val_json datasets\SMB1_LevelsAndCaptions-regular-validate.json --seed %SEED%
call batch\run_diffusion_multi.bat %DIFF_OUTPUT% %TYPE% 
