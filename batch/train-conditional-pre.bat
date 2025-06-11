REM @echo off
REM Usage: train-conditional-pre.bat <seed> <game> <type> <model> [split]
REM <seed> is optional, defaults to 0
REM <game> indicates source of data: SMB1, SMB2, etc.
REM <type> should be "regular", "absence", or "negative"
REM <model> should be "MiniLM" or "GTE"
REM [split] is optional - if "split" is specified, uses split pretrained sentences
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set GAME=%2

set TYPE=%3
if "%TYPE%"=="" set TYPE=regular

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

REM New: Accept model type as final argument (MiniLM or GTE)
set MODEL=%4
if /I "%MODEL%"=="" set MODEL=MiniLM
if /I "%MODEL%"=="MiniLM" set MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
if /I "%MODEL%"=="GTE" set MODEL_NAME=Alibaba-NLP/gte-large-en-v1.5

set SPLIT=%5

if /I "%SPLIT%"=="split" (
    set DIFF_OUTPUT=%GAME%-conditional-%MODEL%split-%TYPE%%SEED%
    set SPLIT_FLAG=--split_pretrained_sentences
) else (
    set DIFF_OUTPUT=%GAME%-conditional-%MODEL%-%TYPE%%SEED%
    set SPLIT_FLAG=
)

set DIFF_FLAGS=
set UNCOND_OUTPUT=%DIFF_OUTPUT%-unconditional-samples

REM Special case for negative prompt training
if /I "%TYPE%"=="negative" (
    set TYPE=regular
    set DIFF_FLAGS=--negative_prompt_training
)

set GAME_PLAYED=
if /I "%GAME%"=="LR" set GAME_PLAYED=--game LR

python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --pretrained_language_model "%MODEL_NAME%" --plot_validation_caption_score --seed %SEED% %DIFF_FLAGS% %SPLIT_FLAG% %DESCRIBE_ABSENCE_FLAG% %GAME_PLAYED%
call batch\run_diffusion_multi.bat %DIFF_OUTPUT% %TYPE% %GAME% text
call batch\evaluate_caption_adherence_multi.bat %DIFF_OUTPUT% %TYPE% %GAME%
