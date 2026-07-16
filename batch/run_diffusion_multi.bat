@echo off
REM Usage: run_diffusion_multi.bat <model_path> <type> <game>
REM <model_path> Directory of trained diffusion model
REM <type> should be "regular", "absence", or "negative"
REM <game> Game model was made for

set MODEL_PATH=%1
set TYPE=%2
set GAME=%3

if "%MODEL_PATH%"=="" (
    echo ERROR: Must provide model_path as first argument.
    exit /b 1
)
if "%TYPE%"=="" set TYPE=regular

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

set UNCOND_OUTPUT=%MODEL_PATH%-unconditional-samples

python run_diffusion.py --model_path %MODEL_PATH% --num_samples 100 --save_as_json --output_dir "%UNCOND_OUTPUT%-short" %DESCRIBE_ABSENCE_FLAG% --game %GAME%
python run_diffusion.py --model_path %MODEL_PATH% --num_samples 100 --save_as_json --output_dir "%UNCOND_OUTPUT%-long" %DESCRIBE_ABSENCE_FLAG% --game %GAME% --level_width 128

