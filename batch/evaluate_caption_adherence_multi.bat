REM @echo off
REM Usage: evaluate_caption_adherence_multi.bat <model_path> <type> <game>
REM <type> should be "regular" or "absence"
REM <game> should be "SMB1", "SMB2", "Mar1and2", etc.
REM This script runs all standard evaluate_caption_adherence.py calls for a given model and type.

set MODEL_PATH=%1
set TYPE=%2
set GAME=%3

if "%MODEL_PATH%"=="" (
    echo ERROR: Must provide model_path as first argument.
    exit /b 1
)
if "%TYPE%"=="" set TYPE=regular

if "%GAME%"=="" set GAME=Mar1and2

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\%GAME%_LevelsAndCaptions-%TYPE%.json --output_dir samples-from-real-%GAME%-captions %DESCRIBE_ABSENCE_FLAG%
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\%GAME%_LevelsAndCaptions-%TYPE%.json --compare_checkpoints %DESCRIBE_ABSENCE_FLAG%
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-test.json --compare_checkpoints %DESCRIBE_ABSENCE_FLAG%
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\%GAME%_RandomTest-%TYPE%.json --output_dir samples-from-random-%GAME%-captions %DESCRIBE_ABSENCE_FLAG%
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\%GAME%_RandomTest-%TYPE%.json --compare_checkpoints %DESCRIBE_ABSENCE_FLAG%

