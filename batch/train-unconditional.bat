@echo off
REM Usage: train-ununconditional.bat <seed> <data> <game>
REM <seed> is optional, defaults to 0
REM <data> indicates source of data: SMB1, SMB2, etc.
REM <game> game to train for
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

set MODEL_DIR=%GAME%-%DATA%-unconditional%SEED%
set UNCOND_OUTPUT=%MODEL_DIR%-samples

set DATA_PATH=Game_%GAME%/DATA/%DATA%_LevelsAndCaptions-regular
set TRAIN_DATA=%DATA_PATH%-train.json
set VAL_DATA=%DATA_PATH%-validate.json

REM Per-execution timing log: each step appends a timestamped record. The log is
REM staged under timing_logs\ during the run then moved into the trained model's directory at the end.
set TIMING_LOG=timing_logs\train-unconditional-%GAME%-%SEED%.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "train-unconditional start"

python train_diffusion.py --augment --output_dir "%MODEL_DIR%" --num_epochs 500 --json %TRAIN_DATA% --val_json %VAL_DATA% --seed %SEED% --game %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion training"

call batch\run_diffusion_multi.bat %MODEL_DIR% regular %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "unconditional samples"

REM move the timing log into the trained model's directory
move /Y %TIMING_LOG% %MODEL_DIR%\pipeline_timing.jsonl