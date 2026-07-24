@echo off
REM Usage: train-conditional-pre.bat <seed> <data> <type> <game> <model> [split]
REM <seed> is optional, defaults to 0
REM <data> indicates source of data: SMB1, SMB2, etc.
REM <type> should be "regular", "absence", or "negative"
REM <game> could be Mario, LR, MM-Simple, etc
REM <model> should be "MiniLM" or "GTE"
REM [split] is optional - if "split" is specified, uses split pretrained sentences
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

REM Accept model type as argument (MiniLM or GTE)
set MODEL=%5
set MODEL_NAME=
if /I "%MODEL%"=="" set MODEL=MiniLM
if /I "%MODEL%"=="MiniLM" set MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
if /I "%MODEL%"=="GTE" set MODEL_NAME=Alibaba-NLP/gte-large-en-v1.5
if /I "%MODEL%"=="CLIP" set MODEL_NAME=sentence-transformers/clip-ViT-L-14
if /I "%MODEL%"=="T5" set MODEL_NAME=google/t5-v1_1-base

if "%MODEL_NAME%"=="" (
    echo Error: Unrecognized model '%MODEL%'.
    exit /b 1
)

set SPLIT=%6

if /I "%SPLIT%"=="split" (
    set DIFF_OUTPUT=%GAME%-%DATA%-conditional-%MODEL%split-%TYPE%%SEED%
    set SPLIT_FLAG=--split_pretrained_sentences
) else (
    set DIFF_OUTPUT=%GAME%-%DATA%-conditional-%MODEL%-%TYPE%%SEED%
    set SPLIT_FLAG=
)

set DIFF_FLAGS=
set UNCOND_OUTPUT=%DIFF_OUTPUT%-unconditional-samples

REM Special case for negative prompt training
if /I "%TYPE%"=="negative" (
    set TYPE=regular
    set DIFF_FLAGS=--negative_prompt_training
)

set DATA_PATH=Game_%GAME%/DATA/%DATA%_LevelsAndCaptions-%TYPE%
set TRAIN_DATA=%DATA_PATH%-train.json
set VAL_DATA=%DATA_PATH%-validate.json
set TEST_DATA=%DATA_PATH%-test.json

REM Per-execution timing log: each step appends a timestamped record. The log is
REM staged under timing_logs\ during the run then moved into the trained model's directory at the end.
set TIMING_LOG=timing_logs\train-conditional-pre-%GAME%-%DATA%-%TYPE%%SEED%.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "train-conditional-pre start"

set DIFFUSION_EPOCHS=500
if "%GAME%"=="LR" set DIFFUSION_EPOCHS=3000

python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs %DIFFUSION_EPOCHS% --json %TRAIN_DATA% --val_json %VAL_DATA% --pretrained_language_model "%MODEL_NAME%" --plot_validation_caption_score --game %GAME% --seed %SEED% %DIFF_FLAGS% %SPLIT_FLAG% %DESCRIBE_ABSENCE_FLAG% 
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion training"

call batch\run_diffusion_multi.bat %DIFF_OUTPUT% %TYPE% %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion samples"

call batch\evaluate_caption_adherence_multi.bat %DIFF_OUTPUT% %TYPE% %DATA% %GAME%
python log_timestamp.py --log_file %TIMING_LOG% --event "caption adherence evaluation"

REM move the timing log into the trained model's directory
move /Y %TIMING_LOG% %DIFF_OUTPUT%\pipeline_timing.jsonl
