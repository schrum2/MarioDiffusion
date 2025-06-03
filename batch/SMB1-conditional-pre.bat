REM @echo off
REM Usage: SMB1-conditional-MiniLM.bat <seed> <type> <model> [split]
REM <seed> is optional, defaults to 0
REM <type> should be "regular", "absence", or "negative"
REM <model> should be "MiniLM" or "GTE"
REM [split] is optional - if "split" is specified, uses split pretrained sentences
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set TYPE=%2
if "%TYPE%"=="" set TYPE=regular

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

REM New: Accept model type as final argument (MiniLM or GTE)
set MODEL=%3
if /I "%MODEL%"=="" set MODEL=MiniLM
if /I "%MODEL%"=="MiniLM" set MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
if /I "%MODEL%"=="GTE" set MODEL_NAME=Alibaba-NLP/gte-large-en-v1.5

set SPLIT=%4

if /I "%SPLIT%"=="split" (
    set DIFF_OUTPUT=SMB1-conditional-%MODEL%split-%TYPE%%SEED%
    set SPLIT_FLAG=--split_pretrained_sentences
) else (
    set DIFF_OUTPUT=SMB1-conditional-%MODEL%-%TYPE%%SEED%
    set SPLIT_FLAG=
)

set DIFF_FLAGS=
set UNCOND_OUTPUT=%DIFF_OUTPUT%-unconditional-samples

REM Special case for negative prompt training
if /I "%TYPE%"=="negative" (
    set TYPE=regular
    set DIFF_FLAGS=--negative_prompt_training
)

python train_diffusion.py --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\SMB1_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\SMB1_LevelsAndCaptions-%TYPE%-validate.json --pretrained_language_model "%MODEL_NAME%" --plot_validation_caption_score --seed %SEED% %DIFF_FLAGS% %SPLIT_FLAG% %DESCRIBE_ABSENCE_FLAG%
call batch\run_diffusion_multi.bat %DIFF_OUTPUT% %TYPE% text
call batch\evaluate_caption_adherence_multi.bat %DIFF_OUTPUT% %TYPE%