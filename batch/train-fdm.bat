REM @echo off
REM Usage: train-fdm.bat <seed> <data> <type> <game> <model>
REM <seed> is optional, defaults to 0
REM <data> indicates source of data: SMB1, SMB2, etc.
REM <type> should be "regular", or "absence"
REM <game> Game to train on: "Mario", "LR", "MMLV", etc
REM <model> should be "MiniLM" or "GTE"
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set DATA=%2

set TYPE=%3
if "%TYPE%"=="" set TYPE=regular

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

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

set MODEL=%5
if /I "%MODEL%"=="" set MODEL=MiniLM
if /I "%MODEL%"=="MiniLM" (
    set MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
    set EMBED_DIM=384
)
if /I "%MODEL%"=="GTE" (
    set MODEL_NAME=Alibaba-NLP/gte-large-en-v1.5
    set EMBED_DIM=1024
)

REM Default values for fdm model output and extra flags
set DIFF_OUTPUT=%GAME%-%DATA%-fdm-%MODEL%-%TYPE%%SEED%

set DATA_PATH=Game_%GAME%/DATA/%DATA%_LevelsAndCaptions-%TYPE%
set TRAIN_DATA=%DATA_PATH%-train.json
set VAL_DATA=%DATA_PATH%-validate.json

python train_fdm.py --augment --output_dir "%DIFF_OUTPUT%" --num_epochs 100 --json %TRAIN_DATA% --val_json %VAL_DATA% --pretrained_language_model "%MODEL_NAME%" --plot_validation_caption_score --embedding_dim "%EMBED_DIM%" --seed %SEED% --game %GAME% %DESCRIBE_ABSENCE_FLAG%
call batch\evaluate_caption_adherence_multi.bat "%DIFF_OUTPUT%" %TYPE% %DATA% %GAME%
