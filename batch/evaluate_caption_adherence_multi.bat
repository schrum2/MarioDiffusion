REM @echo off
REM Usage: evaluate_caption_adherence_multi.bat <model_path> <type> <game>
REM <type> should be "regular" or "absence"
REM <game> should be "SMB1", "SMB2", "Mar1and2", "LR", or "MMLV"
REM This script runs all standard evaluate_caption_adherence.py calls for a given model and type.

set MODEL_PATH=%1
set TYPE=%2
set GAME=%3

if "%MODEL_PATH%"=="" (
    echo ERROR: Must provide model_path as first argument.
    exit /b 1
)
if "%TYPE%"=="" set TYPE=regular

set NUM_TILES=
set GAME_FLAG=

if /I "%GAME%"=="" set GAME=Mar1and2
if /I "%GAME%"=="LR" set NUM_TILES=--num_tiles=8
if /I "%GAME%"=="MMLV" set GAME_FLAG=--game MM-Full

if "%NUM_TILES%"=="" if "%GAME_FLAG%"=="" set NUM_TILES=--num_tiles=13

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence


REM RandomTest captions have no source scene, so randomize the generated width across the
REM training width range. --width_range_json supplies that range for models trained before
REM training_widths.json existed; newer models also carry it in the model directory.
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\%GAME%_RandomTest-%TYPE%.json --output_dir samples-from-random-%GAME%-captions --random_width --width_range_json datasets\%GAME%_LevelsAndCaptions-%TYPE%.json %NUM_TILES% %GAME_FLAG% %DESCRIBE_ABSENCE_FLAG%
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\%GAME%_RandomTest-%TYPE%.json --compare_checkpoints --random_width --width_range_json datasets\%GAME%_LevelsAndCaptions-%TYPE%.json %NUM_TILES% %GAME_FLAG% %DESCRIBE_ABSENCE_FLAG%

REM LevelsAndCaptions captions come from real scenes. Multi-width datasets automatically recreate
REM each caption at its source scene's width; single-width datasets keep the old fixed width.
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\%GAME%_LevelsAndCaptions-%TYPE%.json --output_dir samples-from-real-%GAME%-captions %NUM_TILES% %GAME_FLAG% %DESCRIBE_ABSENCE_FLAG%
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\%GAME%_LevelsAndCaptions-%TYPE%.json --compare_checkpoints %NUM_TILES% %GAME_FLAG% %DESCRIBE_ABSENCE_FLAG%
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-test.json --compare_checkpoints %NUM_TILES% %GAME_FLAG% %DESCRIBE_ABSENCE_FLAG%