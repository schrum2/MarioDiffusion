REM @echo off
REM Usage: evaluate_caption_adherence_multi.bat <model_path> <type>
REM <type> should be "regular" or "absence"
REM This script runs all standard evaluate_caption_adherence.py calls for a given model and type.

set MODEL_PATH=%1
set TYPE=%2

if "%MODEL_PATH%"=="" (
    echo ERROR: Must provide model_path as first argument.
    exit /b 1
)
if "%TYPE%"=="" set TYPE=regular

python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\SMB1_LevelsAndCaptions-%TYPE%.json --output_dir samples-from-real-captions
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\SMB1_LevelsAndCaptions-%TYPE%.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\SMB1_LevelsAndCaptions-%TYPE%-test.json --compare_checkpoints 
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\SMB1_RandomTest-%TYPE%.json --output_dir samples-from-random-captions
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\SMB1_RandomTest-%TYPE%.json --compare_checkpoints
REM python evaluate_metrics.py %MODEL_PATH% %TYPE%
