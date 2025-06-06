REM @echo off
REM Usage: train-fdm.bat <seed> <game> <type> <model>
REM <seed> is optional, defaults to 0
REM <game> indicates source of data: SMB1, SMB2, etc.
REM <type> should be "regular", or "absence"
REM <model> should be "MiniLM" or "GTE"
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set GAME=%2

set TYPE=%3
if "%TYPE%"=="" set TYPE=regular

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

set MODEL=%4
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
set DIFF_OUTPUT=%GAME%-fdm-%MODEL%-%TYPE%%SEED%


python train_fdm.py --augment --output_dir "%DIFF_OUTPUT%" --num_epochs 100 --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --pretrained_language_model "%MODEL_NAME%" --plot_validation_caption_score --embedding_dim "%EMBED_DIM%" --seed %SEED% %DESCRIBE_ABSENCE_FLAG%
call batch\evaluate_caption_adherence_multi.bat "%DIFF_OUTPUT%\\final-model" %TYPE% %GAME%
