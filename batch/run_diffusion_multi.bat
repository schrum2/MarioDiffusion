REM @echo off
REM Usage: run_diffusion_multi.bat <model_path> <type> <text>
REM <type> should be "regular", "absence", or "negative"
REM This script runs all standard evaluate_caption_adherence.py calls for a given model and type.

set MODEL_PATH=%1
set TYPE=%2
set TEXT=%3

if "%MODEL_PATH%"=="" (
    echo ERROR: Must provide model_path as first argument.
    exit /b 1
)
if "%TYPE%"=="" set TYPE=regular

set TEXT_FLAG=
if /I "%TEXT%"=="text" set TEXT_FLAG=--text_conditional

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

set UNCOND_OUTPUT=%MODEL_PATH%-unconditional-samples

python run_diffusion.py --model_path %MODEL_PATH% --num_samples 100 %TEXT_FLAG% --save_as_json --output_dir "%UNCOND_OUTPUT%-short" %DESCRIBE_ABSENCE_FLAG%
python run_diffusion.py --model_path %MODEL_PATH% --num_samples 100 %TEXT_FLAG% --save_as_json --output_dir "%UNCOND_OUTPUT%-long" %DESCRIBE_ABSENCE_FLAG% --level_width 128

