REM @echo off
REM Usage: train-fdm.bat <seed> <game> <type> <model>
REM <seed> is optional, defaults to 0
REM <game> indicates source of data: SMB1, SMB2, etc.
REM <type> should be "regular", or "absence"
REM <model> should be "MiniLM" or "GTE"
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set TYPE=%2
if "%TYPE%"=="" set TYPE=regular

REM Add --describe_absence flag if TYPE is absence
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

set MODEL=%3
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
set DIFF_OUTPUT=Mar1and2-fdm-%MODEL%-%TYPE%%SEED%


python temp_find_best_adherance_score_for_fdm.py --model_path %DIFF_OUTPUT%
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json datasets\Mar1and2_RandomTest-%TYPE%.json --output_dir samples-from-random-Mar1and2-captions %DESCRIBE_ABSENCE_FLAG%
python evaluate_caption_adherence.py --model_path %DIFF_OUTPUT% --save_as_json --json datasets\Mar1and2_LevelsAndCaptions-%TYPE%.json --output_dir samples-from-real-Mar1and2-captions %DESCRIBE_ABSENCE_FLAG%

