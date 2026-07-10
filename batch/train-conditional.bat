@echo off
REM Usage: train-conditional.bat <seed> <game> <type> 
REM <seed> is optional, defaults to 0
REM <game> indicates source of data: SMB1, SMB2, etc.
REM <type> should be "regular", "absence", or "negative"
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set GAME=%2

set TYPE=%3
if "%TYPE%"=="" set TYPE=regular

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

REM Set up variables for all cases
set MLM_OUTPUT=%GAME%-MLM-%TYPE%%SEED%

REM Default values for conditional model output and extra flags
set MODEL_DIR=%GAME%-conditional-%TYPE%%SEED%
set DIFF_FLAGS=

REM Special case for negative prompt training
if /I "%TYPE%"=="negative" (
    set TYPE=regular
    set DIFF_FLAGS=--negative_prompt_training
)

set GAME_PLAYED=
if /I "%GAME%"=="LR" set GAME_PLAYED=--game LR
else set GAME_PLAYED=--game Mario

REM Per-execution timing log: each step appends a timestamped record. The log is
REM staged under timing_logs\ during the run then moved into the trained model's directory at the end.
set TIMING_LOG=timing_logs\train-conditional-%GAME%-%TYPE%-%SEED%.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "train-conditional start"

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --test_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-test.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --output_dir %MLM_OUTPUT% --seed %SEED%
python log_timestamp.py --log_file %TIMING_LOG% --event "MLM training"

python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir "%MODEL_DIR%" --num_epochs 500 --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% %DIFF_FLAGS% %DESCRIBE_ABSENCE_FLAG% %GAME_PLAYED%
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion training"

call batch\run_diffusion_multi.bat %MODEL_DIR% %TYPE% %GAME% text
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion samples"

call batch\evaluate_caption_adherence_multi.bat %MODEL_DIR% %TYPE% %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "caption adherence evaluation"

REM move the timing log into the trained model's directory
move /Y %TIMING_LOG% %MODEL_DIR%\pipeline_timing.jsonl
