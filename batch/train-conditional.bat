@echo off
REM Usage: train-conditional.bat <seed> <data> <type> <game>
REM <seed> is optional, defaults to 0
REM <data> indicates source of data: SMB1, SMB2, Mar1and2, LR, etc.
REM <type> should be "regular", "absence", or "negative"
REM <game> defaults to "Mario" but could be "LR", "MM2", etc.
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set DATA=%2

set TYPE=%3
if "%TYPE%"=="" set TYPE=regular

set GAME=%4

set "VALID=false"
for %%G in (Mario LR MM-Simple MM-Full MMLV MM2) do (
    if /I "%GAME%"=="%%~G" set "VALID=true"
)

REM Exit if the flag was never flipped to true
if "%VALID%"=="false" (
    echo Error: Invalid game selected.
    exit /b 1
)

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

REM Set up variables for all cases
set MLM_OUTPUT=%GAME%-%DATA%-MLM-%TYPE%%SEED%

REM Default values for conditional model output and extra flags
set MODEL_DIR=%GAME%-%DATA%-conditional-%TYPE%%SEED%
set DIFF_FLAGS=

REM Special case for negative prompt training
if /I "%TYPE%"=="negative" (
    set TYPE=regular
    set DIFF_FLAGS=--negative_prompt_training
)

set DATA_PATH=Game_%GAME%/DATA/%DATA%_LevelsAndCaptions-%TYPE%
set TRAIN_DATA=%DATA_PATH%-train.json
set VAL_DATA=%DATA_PATH%-validate.json
set TEST_DATA=%DATA_PATH%-test.json
set TOKENIZER=Game_%GAME%/DATA/%DATA%_Tokenizer-%TYPE%.pkl

REM Per-execution timing log: each step appends a timestamped record. The log is
REM staged under timing_logs\ during the run then moved into the trained model's directory at the end.
set TIMING_LOG=timing_logs\train-conditional-%GAME%-%DATA%-%TYPE%%SEED%.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "train-conditional start"

set MLM_EPOCHS=300
set MLM_CHECKPOINT=20
if "%GAME%"=="LR" (
    REM Is 80,000 really correct?
    set MLM_EPOCHS=80000
    set MLM_CHECKPOINT=1000
)

python train_mlm.py --epochs %MLM_EPOCHS% --checkpoint_freq %MLM_CHECKPOINT% --save_checkpoints --json %TRAIN_DATA% --val_json %VAL_DATA% --test_json %TEST_DATA% --pkl %TOKENIZER% --output_dir %MLM_OUTPUT% --seed %SEED%
python log_timestamp.py --log_file %TIMING_LOG% --event "MLM training"

set DIFFUSION_EPOCHS=500
if "%GAME%"=="LR" (
    set GAME=LR
    set DIFFUSION_EPOCHS=3000
)

python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir "%MODEL_DIR%" --num_epochs %DIFFUSION_EPOCHS% --json %TRAIN_DATA% --val_json %VAL_DATA% --pkl %TOKENIZER% --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% %DIFF_FLAGS% %DESCRIBE_ABSENCE_FLAG% --game %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion training"

call batch\run_diffusion_multi.bat %MODEL_DIR% %TYPE% %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion samples"

call batch\evaluate_caption_adherence_multi.bat %MODEL_DIR% %TYPE% %DATA% %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "caption adherence evaluation"

REM move the timing log into the trained model's directory
move /Y %TIMING_LOG% %MODEL_DIR%\pipeline_timing.jsonl
