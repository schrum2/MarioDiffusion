@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM MMLV-complete.bat - full MMLV pipeline: download -> ASCII -> scenes ->
REM captions -> split -> train -> evaluate. Run from anywhere.
REM
REM Usage: MMLV-complete.bat [seed] [data] [type] [model] [split] [diffusion_epochs] [lvl_target] [post_evaluation]
REM
REM   [seed]   optional, defaults to 0
REM   [data]   optional dataset tag, defaults to today's date (MM_dd, e.g. 11_13).
REM            All files are named after it:
REM              Game_MMLV\DATA\<data>_MMLV_Levels\            (ASCII levels)
REM              Game_MMLV\DATA\<data>_MMLV_Levels.json        (scenes)
REM              Game_MMLV\DATA\<data>_MMLV_LevelsAndCaptions-<type>[-train|-validate|-test].json
REM              Game_MMLV\MMLV_<data>-conditional-<model>-<split>-<type>-seed<seed>\  (model)
REM   [type]   optional, defaults to "regular". One of "regular", "absence",
REM            or "negative" (regular captions + --negative_prompt_training)
REM   [model]  optional pretrained text encoder, defaults to "CLIP". One of:
REM              CLIP    - sentence-transformers/clip-ViT-L-14
REM              MiniLM  - sentence-transformers/multi-qa-MiniLM-L6-cos-v1
REM              GTE     - Alibaba-NLP/gte-large-en-v1.5
REM              T5      - google/t5-v1_1-base
REM            (MLM is not supported here: no local MLM/tokenizer training.)
REM   [split]  optional, defaults to "single". "single" encodes the whole caption
REM            with one vector, "multiple" encodes each phrase separately.
REM   [diffusion_epochs] optional, defaults to 500.
REM   [lvl_target] optional, defaults to 5000. Number of valid MMLV levels the
REM            bulk download aims for (levels already downloaded to
REM            %%LOCALAPPDATA%%\MegaMaker\Levels are reused). 0 skips the download.
REM   [post_evaluation] optional, defaults to "yes". "no" skips sample
REM            generation, caption adherence, and prompt/level pairs.
REM
REM Data steps (ASCII conversion, scenes, captions, split) are skipped when their
REM output already exists, so re-running with the same [data] tag reuses them.
REM Delete the outputs (or pick a new tag) to rebuild. Training resumes if the
REM model directory already exists.
REM
REM Stops at the first failing step. Per-step logs go to Game_MMLV\logs_<data>\.
REM Set PY beforehand to use a different interpreter.
REM ============================================================================

cd /d "%~dp0..\.."
if "%PY%"=="" set PY=C:\Users\wilsonf\AppData\Local\miniconda3\envs\SURF\python.exe
set PYTHONUNBUFFERED=1
set GAME=MMLV

REM --- Scene-extraction settings (create_megaman_json_data.py) ---------------
set SCENE_LIMIT=50000
set MAX_ENEMIES=10
set MIN_CONTENT_PCT=4
set SCENE_FLAGS=--include_teleporters

REM --- Read arguments --------------------------------------------------------
set SEED=%~1
if "%SEED%"=="" set SEED=0

set DATA=%~2
if "%DATA%"=="" for /f %%D in ('powershell -NoProfile -Command "Get-Date -Format MM_dd"') do set DATA=%%D

set TYPE=%~3
if "%TYPE%"=="" set TYPE=regular

set MODEL=%~4
if "%MODEL%"=="" set MODEL=CLIP

set SPLIT=%~5
if "%SPLIT%"=="" set SPLIT=single

set DIFFUSION_EPOCHS=%~6
if "%DIFFUSION_EPOCHS%"=="" set DIFFUSION_EPOCHS=500

set LVL_TARGET=%~7
if "%LVL_TARGET%"=="" set LVL_TARGET=5000

set POST_EVALUATION=%~8
if "%POST_EVALUATION%"=="" set POST_EVALUATION=yes

REM --- Validate --------------------------------------------------------------
set "TYPE_VALID=false"
for %%T in (regular absence negative) do if /I "%TYPE%"=="%%T" set "TYPE_VALID=true"
if "%TYPE_VALID%"=="false" (
    echo Error: Invalid type '%TYPE%'. Must be regular, absence, or negative.
    exit /b 1
)

set MODEL_NAME=
if /I "%MODEL%"=="CLIP" set MODEL_NAME=sentence-transformers/clip-ViT-L-14
if /I "%MODEL%"=="MiniLM" set MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
if /I "%MODEL%"=="GTE" set MODEL_NAME=Alibaba-NLP/gte-large-en-v1.5
if /I "%MODEL%"=="T5" set MODEL_NAME=google/t5-v1_1-base
if "%MODEL_NAME%"=="" (
    echo Error: Unrecognized model '%MODEL%'. Must be CLIP, MiniLM, GTE, or T5. MLM is not supported here.
    exit /b 1
)

if /I not "%SPLIT%"=="single" if /I not "%SPLIT%"=="multiple" (
    echo Error: Invalid split '%SPLIT%'. Must be "single" or "multiple".
    exit /b 1
)

if /I not "%POST_EVALUATION%"=="yes" if /I not "%POST_EVALUATION%"=="no" (
    echo Error: Invalid post_evaluation '%POST_EVALUATION%'. Must be "yes" or "no".
    exit /b 1
)

REM --- Type-specific flags ---------------------------------------------------
REM "negative" uses the regular captions plus negative prompt training.
set CAPTION_TYPE=%TYPE%
set CAPTION_FLAGS=
set DIFF_FLAGS=
if /I "%TYPE%"=="absence" set CAPTION_FLAGS=--describe_absence
if /I "%TYPE%"=="negative" (
    set CAPTION_TYPE=regular
    set DIFF_FLAGS=--negative_prompt_training
)

set SPLIT_FLAG=
if /I "%SPLIT%"=="multiple" set SPLIT_FLAG=--split_pretrained_sentences

REM --- Paths -----------------------------------------------------------------
set PREFIX=Game_MMLV\DATA\%DATA%_MMLV
set LEVELS_DIR=%PREFIX%_Levels
set LEVELS_JSON=%PREFIX%_Levels.json
set CAPTIONS=%PREFIX%_LevelsAndCaptions-%CAPTION_TYPE%
set MODEL_DIR=Game_MMLV\MMLV_%DATA%-conditional-%MODEL%-%SPLIT%-%TYPE%-seed%SEED%
set LOGS=Game_MMLV\logs_%DATA%
if not exist Game_MMLV\DATA mkdir Game_MMLV\DATA
if not exist %LOGS% mkdir %LOGS%

echo Data tag:  %DATA%
echo Model dir: %MODEL_DIR%
echo Logs:      %LOGS%

REM ===========================================================================
REM Step 1: bulk download MMLV levels (to %LOCALAPPDATA%\MegaMaker\Levels)
REM ===========================================================================
set NAME=1_download
if "%LVL_TARGET%"=="0" (
    call :skip "lvl_target is 0"
) else (
    call :start
    "%PY%" Game_MMLV\bulk_mmlv_download.py --target %LVL_TARGET% > %LOGS%\!NAME!.log 2>&1 || goto :fail
)

REM ===========================================================================
REM Step 2: convert downloaded MMLV levels to ASCII
REM ===========================================================================
set NAME=2_ascii
if exist "%LEVELS_DIR%\" (
    call :skip "%LEVELS_DIR% exists"
) else (
    call :start
    "%PY%" Game_MMLV\bulk_mmlv_to_vglc.py --output %LEVELS_DIR% > %LOGS%\!NAME!.log 2>&1 || goto :fail
)

REM ===========================================================================
REM Step 3: cut levels into scenes (create_megaman_json_data.py)
REM ===========================================================================
set NAME=3_scenes
if exist "%LEVELS_JSON%" (
    call :skip "%LEVELS_JSON% exists"
) else (
    call :start
    "%PY%" create_megaman_json_data.py --limit %SCENE_LIMIT% --levels %LEVELS_DIR% --tileset Game_MMLV\MMLV.json --stride_x 16 --stride_y 14 --scan_mode screen_grid --output %LEVELS_JSON% --max_enemies %MAX_ENEMIES% --min_content_pct %MIN_CONTENT_PCT% %SCENE_FLAGS% > %LOGS%\!NAME!.log 2>&1 || goto :fail
)

REM ===========================================================================
REM Step 4: deterministic captions
REM ===========================================================================
set NAME=4_captions_%CAPTION_TYPE%
if exist "%CAPTIONS%.json" (
    call :skip "%CAPTIONS%.json exists"
) else (
    call :start
    "%PY%" MMLV_create_ascii_captions.py --dataset %LEVELS_JSON% --tileset Game_MMLV\MMLV.json --output %CAPTIONS%.json %CAPTION_FLAGS% > %LOGS%\!NAME!.log 2>&1 || goto :fail
)

REM ===========================================================================
REM Step 5: train/validate/test split (90/5/5)
REM ===========================================================================
set NAME=5_split_%CAPTION_TYPE%
if exist "%CAPTIONS%-train.json" (
    call :skip "%CAPTIONS%-train.json exists"
) else (
    call :start
    "%PY%" split_data.py --json_file %CAPTIONS%.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game %GAME% > %LOGS%\!NAME!.log 2>&1 || goto :fail
)

REM ===========================================================================
REM Step 6: diffusion training (resumes if MODEL_DIR exists)
REM ===========================================================================
set NAME=6_train-%MODEL%-%SPLIT%-%TYPE%-seed%SEED%
call :start
"%PY%" train_diffusion.py --save_image_epochs %DIFFUSION_EPOCHS% --augment --text_conditional --output_dir %MODEL_DIR% --num_epochs %DIFFUSION_EPOCHS% --json %CAPTIONS%-train.json --val_json %CAPTIONS%-validate.json --seed %SEED% --game %GAME% %DIFF_FLAGS% %CAPTION_FLAGS% --plot_validation_caption_score --plot_clip_score --pretrained_language_model "%MODEL_NAME%" %SPLIT_FLAG% > %LOGS%\%NAME%.log 2>&1 || goto :fail

if /I "%POST_EVALUATION%"=="no" goto :done

REM ===========================================================================
REM Step 7: unconditional samples (saved to model dir)
REM ===========================================================================
set NAME=7_unconditional-%MODEL%-%SPLIT%-%TYPE%-seed%SEED%
call :start
"%PY%" run_diffusion.py --model_path %MODEL_DIR% --num_samples 100 --save_as_json --output_dir %MODEL_DIR%\unconditional-samples-short --game %GAME% > %LOGS%\%NAME%.log 2>&1 || goto :fail

REM ===========================================================================
REM Step 8: caption adherence on test split (saved to model dir)
REM ===========================================================================
set NAME=8_adherence-%MODEL%-%SPLIT%-%TYPE%-seed%SEED%
call :start
"%PY%" evaluate_caption_adherence.py --model_path %MODEL_DIR% --save_as_json --json %CAPTIONS%-test.json --output_dir samples-from-test-captions --game %GAME% %CAPTION_FLAGS% > %LOGS%\%NAME%.log 2>&1 || goto :fail

REM ===========================================================================
REM Step 9: prompt/level pairs (saved to model dir)
REM ===========================================================================
set NAME=9_prompt_pairs-%MODEL%-%SPLIT%-%TYPE%-seed%SEED%
call :start
"%PY%" generate_prompt_level_pairs.py --model_path %MODEL_DIR% --captions_json %CAPTIONS%-test.json --num_pairs 50 --game %GAME% --save_images > %LOGS%\%NAME%.log 2>&1 || goto :fail

:done
echo [%date% %time%] ALL STEPS DONE
echo [%date% %time%] ALL STEPS DONE>> %LOGS%\_progress.log
exit /b 0

:start
echo [%date% %time%] START %NAME%
echo [%date% %time%] START %NAME%>> %LOGS%\_progress.log
exit /b 0

:skip
echo [%date% %time%] SKIP %NAME% (%~1)
echo [%date% %time%] SKIP %NAME% (%~1)>> %LOGS%\_progress.log
exit /b 0

:fail
echo [%date% %time%] FAILED at %NAME% - see %LOGS%\%NAME%.log
echo [%date% %time%] FAILED at %NAME%>> %LOGS%\_progress.log
exit /b 1
