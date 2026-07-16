@echo off
REM Usage: evaluate_caption_adherence_multi.bat <model_path> <type> <data> <game>
REM <type> should be "regular" or "absence"
REM <data> Dataset prefix: should be "SMB1", "SMB2", "Mar1and2", "LR", etc
REM <game> Game: Mario, LR, MMLV, etc.
REM This script runs all standard evaluate_caption_adherence.py calls for a given model and type.

set MODEL_PATH=%1
set TYPE=%2
set DATA=%3
set GAME=%4

if "%MODEL_PATH%"=="" (
    echo ERROR: Must provide model_path as first argument.
    exit /b 1
)
if "%TYPE%"=="" set TYPE=regular

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

set DATA_PREFIX=Game_%GAME%/DATA/%DATA%
set DATA_PATH=%DATA_PREFIX%_LevelsAndCaptions-%TYPE%
set TEST_DATA=%DATA_PATH%-test.json
set TOKENIZER=Game_%GAME%/DATA/%DATA%_Tokenizer-%TYPE%.pkl

REM RandomTest captions have no source scene, so randomize the generated width across the
REM training width range. --width_range_json supplies that range for models trained before
REM training_widths.json existed; newer models also carry it in the model directory.
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json %DATA_PREFIX%_RandomTest-%TYPE%.json --output_dir samples-from-random-%DATA%-captions --random_width --width_range_json %DATA_PREFIX%_LevelsAndCaptions-%TYPE%.json --game %GAME% %DESCRIBE_ABSENCE_FLAG%
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json %DATA_PREFIX%_RandomTest-%TYPE%.json --compare_checkpoints --random_width --width_range_json %DATA_PREFIX%_LevelsAndCaptions-%TYPE%.json --game %GAME% %DESCRIBE_ABSENCE_FLAG%

REM LevelsAndCaptions captions come from real scenes. Multi-width datasets automatically recreate
REM each caption at its source scene's width; single-width datasets keep the old fixed width.
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json %DATA_PATH%.json --output_dir samples-from-real-%DATA%-captions --game %GAME% %DESCRIBE_ABSENCE_FLAG%
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json %DATA_PATH%.json --compare_checkpoints --game %GAME% %DESCRIBE_ABSENCE_FLAG%
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json %TEST_DATA% --compare_checkpoints --game %GAME% %DESCRIBE_ABSENCE_FLAG%