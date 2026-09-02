@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM train-diffusion.bat - unified training entry point
REM
REM Usage: train-diffusion.bat <seed> <data> <type> <game> [model] [split] [tile_embed_method] [tile_embed_dim] [diffusion_epochs] [num_captions] [extra args...]
REM
REM   <seed>   optional, defaults to 0
REM   <data>   source of data: SMB1, SMB2, Mar1and2, LR, etc.
REM   <type>   "regular", "absence", "negative", "llm", or "none"
REM              - "none" trains an UNCONDITIONAL model (no captions at all)
REM              - "negative" trains a regular conditional model with
REM                --negative_prompt_training enabled
REM              - "llm" should be combined with [extra args...] to 
REM                identify caption sources
REM   <game>   Mario, LR, MM-Simple, MM-Full, MMLV, MM2
REM   [model]  optional, defaults to "MLM". One of:
REM              MLM     - trains its own MLM transformer text encoder
REM              MiniLM  - sentence-transformers/multi-qa-MiniLM-L6-cos-v1
REM              GTE     - Alibaba-NLP/gte-large-en-v1.5
REM              CLIP    - sentence-transformers/clip-ViT-L-14
REM              T5      - google/t5-v1_1-base
REM            Ignored (and meaningless) when <type> is "none".
REM   [split]  optional, defaults to "single". One of "single" or "multiple".
REM              Only meaningful when using a pretrained text encoder (i.e.
REM              [model] is not "MLM" and <type> is not "none"). "multiple"
REM              means each caption phrase is encoded separately; "single"
REM              encodes the whole caption with one vector
REM   [tile_embed_method] optional, defaults to "none". One of:
REM              none      - no tile embedding model; tiles use the default
REM                          one-hot encoding
REM              block2vec - train (or reuse) a block2vec tile embedding model
REM              skip      - train (or reuse) a skip-gram tile embedding model
REM              Can be combined with any of the text-encoder options above
REM              (MLM or any pretrained model), or used with unconditional
REM              training.
REM   [tile_embed_dim] optional, defaults to 16. Embedding dimension used when
REM              [tile_embed_method] is not "none". Ignored otherwise.
REM   [diffusion_epochs] optional, defaults to 500. Number of epochs used for
REM              diffusion-model training.
REM   [num_captions] optional. Assumes [extra args...] will be specified. For
REM              each caption source key, only sample this many captions from data.
REM   [extra args...] optional. Any additional arguments are treated as
REM              caption_source_keys values and forwarded to train_diffusion.py
REM              as --caption_source_keys <key1> <key2> ... . If supplied,
REM              the run must use a non-MLM, non-unconditional training setup.
REM              These runs also skip caption-adherence evaluation.
REM
REM Data sources containing "128" use --batch_size 16 because their larger
REM training samples need a smaller batch size to fit in VRAM.
REM
REM ============================================================================
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set DATA=%2
if "%DATA%"=="" (
    echo Error: No data source selected.
    exit /b 1
)

set TYPE=%3
if "%TYPE%"=="" (
    echo Error: No caption type selected. Choose: "regular", "absence", "negative", "llm", or "none"
    exit /b 1
)

set GAME=%4

REM --- Validate GAME -----------------------------------------------------
set "VALID=false"
for %%G in (Mario LR MM-Simple MM-Full MMLV MM2) do (
    if /I "%GAME%"=="%%~G" set "VALID=true"
)
if "%VALID%"=="false" (
    echo Error: Invalid game selected.
    exit /b 1
)

REM --- Resolve game-specific data directory --------------------------------
REM Mega Man variants share the same dataset layout and README, but they may
REM use different tilesets at runtime. Keep the game alias for scripts and
REM training names, but point data lookups at the shared Mega Man directory.
set GAME_DIR=Game_%GAME%
if /I "%GAME%"=="MM-Simple" set GAME_DIR=Game_MM
if /I "%GAME%"=="MM-Full" set GAME_DIR=Game_MM

REM --- Validate TYPE -------------------------------------------------------
set "TYPE_VALID=false"
for %%T in (regular absence negative llm none) do (
    if /I "%TYPE%"=="%%~T" set "TYPE_VALID=true"
)
if "%TYPE_VALID%"=="false" (
    echo Error: Invalid type selected. Must be regular, absence, negative, llm, or none.
    exit /b 1
)

REM --- Read MODEL / SPLIT ---------------------------------------------------
set MODEL=%5
if "%MODEL%"=="" set MODEL=MLM

set SPLIT=%6
if "%SPLIT%"=="" set SPLIT=single
if /I not "%SPLIT%"=="single" if /I not "%SPLIT%"=="multiple" (
    echo Error: Invalid split value '%SPLIT%'. Must be "single" or "multiple".
    exit /b 1
)

REM --- Read tile embedding options --------------------------------------
set TILE_METHOD=%7
if "%TILE_METHOD%"=="" set TILE_METHOD=none

set TILE_DIM=%8
if "%TILE_DIM%"=="" set TILE_DIM=16

REM --- Read diffusion training epochs -----------------------------------
set DIFFUSION_EPOCHS=%9
if "%DIFFUSION_EPOCHS%"=="" set DIFFUSION_EPOCHS=500

REM --- Read optional captions-per-key limit for caption-source pools ------
set NUM_CAPTIONS=%~10
if "%NUM_CAPTIONS%"=="" set NUM_CAPTIONS=
if defined NUM_CAPTIONS (
    set /a NUM_CAPTIONS_CHECK=%NUM_CAPTIONS% 2>nul
    if errorlevel 1 (
        echo Error: Invalid num_captions '%NUM_CAPTIONS%'. Must be a positive integer.
        exit /b 1
    )
    if %NUM_CAPTIONS% LSS 1 (
        echo Error: Invalid num_captions '%NUM_CAPTIONS%'. Must be a positive integer.
        exit /b 1
    )
)

REM --- Read caption_source_key values from any extra parameters ---------
set "CAPTION_SOURCE_KEYS="
set "CAPTION_SOURCE_KEYS_ARG="
set "CAPTIONS_PER_KEY_FLAG="
if defined NUM_CAPTIONS set "CAPTIONS_PER_KEY_FLAG=--captions_per_key %NUM_CAPTIONS%"
shift
shift
shift
shift
shift
shift
shift
shift
shift
shift
:parse_caption_source_keys
if "%~1"=="" goto end_parse_caption_source_keys
if /I "%~1"=="--caption_source_keys" (
    shift
    if "%~1"=="" goto end_parse_caption_source_keys
)
set "CAPTION_SOURCE_KEYS=!CAPTION_SOURCE_KEYS! %~1"
shift
goto parse_caption_source_keys
:end_parse_caption_source_keys
if defined CAPTION_SOURCE_KEYS (
    set "CAPTION_SOURCE_KEYS_ARG=--caption_source_keys !CAPTION_SOURCE_KEYS:~1!"
)
if defined NUM_CAPTIONS if not defined CAPTION_SOURCE_KEYS (
    echo Error: num_captions requires caption_source_keys values in the extra args.
    exit /b 1
)

set "TILE_VALID=false"
for %%E in (none block2vec skip) do (
    if /I "%TILE_METHOD%"=="%%~E" set "TILE_VALID=true"
)
if "%TILE_VALID%"=="false" (
    echo Error: Invalid tile embedding method '%TILE_METHOD%'. Must be none, block2vec, or skip.
    exit /b 1
)

set USE_TILE_EMBED=false
if /I not "%TILE_METHOD%"=="none" set USE_TILE_EMBED=true

REM --- Decide which text-conditioning pipeline this run is -----------------
REM UNCONDITIONAL = true  -> type is "none", no captions/text encoder at all
REM USE_MLM       = true  -> train our own MLM transformer text encoder
REM otherwise             -> use a pretrained text embedding model
set UNCONDITIONAL=false
if /I "%TYPE%"=="none" set UNCONDITIONAL=true

set USE_MLM=false
set MODEL_NAME=
if /I "%UNCONDITIONAL%"=="false" (
    if /I "%MODEL%"=="MLM" (
        set USE_MLM=true
    ) else (
        if /I "%MODEL%"=="MiniLM" set MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
        if /I "%MODEL%"=="GTE" set MODEL_NAME=Alibaba-NLP/gte-large-en-v1.5
        if /I "%MODEL%"=="CLIP" set MODEL_NAME=sentence-transformers/clip-ViT-L-14
        if /I "%MODEL%"=="T5" set MODEL_NAME=google/t5-v1_1-base
        if "!MODEL_NAME!"=="" (
            echo "Error: Unrecognized model '%MODEL%'."
            exit /b 1
        )
    )
)

if defined CAPTION_SOURCE_KEYS (
    if /I "%UNCONDITIONAL%"=="true" (
        echo "Error: caption_source_keys cannot be used with unconditional training (type none)."
        exit /b 1
    )
    if /I "%USE_MLM%"=="true" (
        echo "Error: caption_source_keys cannot be used with model MLM."
        exit /b 1
    )
)

REM --- describe_absence flag -------------------------------------------------
set DESCRIBE_ABSENCE_FLAG=
if /I "%TYPE%"=="absence" set DESCRIBE_ABSENCE_FLAG=--describe_absence

REM --- negative prompt training special case ---------------------------------
set DIFF_FLAGS=
if /I "%TYPE%"=="negative" (
    set TYPE=regular
    set DIFF_FLAGS=--negative_prompt_training
)

REM --- split flag: only applies to pretrained text encoders -------------------
set SPLIT_FLAG=
if /I "%UNCONDITIONAL%"=="false" (
    if /I "%USE_MLM%"=="false" (
        if /I "%SPLIT%"=="multiple" (
            set SPLIT_FLAG=--split_pretrained_sentences
        )
    )
)

REM --- Tile embedding dataset / output dir / naming tag -----------------
set TILE_JSON=
set EMBEDDING_DIR=
set BLOCK_EMBED_FLAG=
set TILE_TAG=
if /I "%USE_TILE_EMBED%"=="true" (
    set TILE_JSON=%GAME_DIR%/DATA/%DATA%_3x3_tiles.json
    if not exist "!TILE_JSON!" (
        echo Error: Tile-level dataset "!TILE_JSON!" does not exist. Is needed to train tile-embedding model.
        exit /b 1
    )
    set EMBEDDING_DIR=%GAME%-%DATA%-%TILE_METHOD%%TILE_DIM%-embeddings-seed%SEED%
    set BLOCK_EMBED_FLAG=--block_embedding_model_path "!EMBEDDING_DIR!"
    set TILE_TAG=%TILE_METHOD%%TILE_DIM%
)

REM --- Data paths --------------------------------------------------------
if /I "%UNCONDITIONAL%"=="true" (
    set DATA_PATH=%GAME_DIR%/DATA/%DATA%_LevelsAndCaptions-regular
) else (
    set DATA_PATH=%GAME_DIR%/DATA/%DATA%_LevelsAndCaptions-%TYPE%
)
set TRAIN_DATA=%DATA_PATH%-train.json
set VAL_DATA=%DATA_PATH%-validate.json
set TEST_DATA=%DATA_PATH%-test.json

REM --- Output directory naming ------------------------------------------
set MLM_OUTPUT=
set TOKENIZER=
set CAPTION_LIMIT_TAG=
if defined NUM_CAPTIONS if /I "%UNCONDITIONAL%"=="false" set "CAPTION_LIMIT_TAG=-captions%NUM_CAPTIONS%"
if /I "%UNCONDITIONAL%"=="true" (
    if /I "%USE_TILE_EMBED%"=="true" (
        set MODEL_DIR=%GAME%-%DATA%-unconditional-%TILE_TAG%-seed%SEED%
    ) else (
        set MODEL_DIR=%GAME%-%DATA%-unconditional-seed%SEED%
    )
) else (
    if /I "%USE_MLM%"=="true" (
        set MLM_OUTPUT=%GAME%-%DATA%-MLM-%TYPE%-seed%SEED%
        set TOKENIZER=%GAME_DIR%/DATA/%DATA%_Tokenizer-%TYPE%.pkl
        if /I "%USE_TILE_EMBED%"=="true" (
            set MODEL_DIR=%GAME%-%DATA%-conditional-%TILE_TAG%-%TYPE%!CAPTION_LIMIT_TAG!-seed%SEED%
        ) else (
            set MODEL_DIR=%GAME%-%DATA%-conditional-%TYPE%!CAPTION_LIMIT_TAG!-seed%SEED%
        )
    ) else (
        set MODEL_TAG=%MODEL%-%SPLIT%
        if /I "%USE_TILE_EMBED%"=="true" (
            set MODEL_DIR=%GAME%-%DATA%-conditional-!MODEL_TAG!-%TILE_TAG%-%TYPE%!CAPTION_LIMIT_TAG!-seed%SEED%
        ) else (
            set MODEL_DIR=%GAME%-%DATA%-conditional-!MODEL_TAG!-%TYPE%!CAPTION_LIMIT_TAG!-seed%SEED%
        )
    )
)

REM --- Per-execution timing log -------------------------------------------
REM Each step appends a timestamped record. The log is staged under
REM timing_logs\ during the run then moved into the trained model's
REM directory at the end.
set TIMING_LOG=timing_logs\train-%GAME%-%DATA%-%TYPE%-seed%SEED%.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "train start"

REM --- MLM epoch counts -----------------------------------------------------
set MLM_EPOCHS=300
set MLM_CHECKPOINT=20
set MLM_MAX_SEQ_LENGTH_FLAG=
if /I "%GAME%"=="MM2" set MLM_MAX_SEQ_LENGTH_FLAG=--max_seq_length 200
if /I "%GAME%"=="LR" (
    REM Is 60,000 really correct?
    set MLM_EPOCHS=60000
    set MLM_CHECKPOINT=1000
)

REM --- 128-size data batch-size adjustment --------------------------------
REM Data sources with "128" in the name use larger training samples, which
REM need a smaller batch size to fit in VRAM (on this hardware, anyway).
set BATCH_SIZE_FLAG=
echo %DATA%| findstr /C:"128" >nul
if %ERRORLEVEL% EQU 0 (
    set BATCH_SIZE_FLAG=--batch_size 16
)

REM ===========================================================================
REM Step 0: train (or reuse) a tile embedding model, if requested.
REM If EMBEDDING_DIR already exists, training is skipped and the existing
REM model is reused as-is.
REM ===========================================================================
if /I "%USE_TILE_EMBED%"=="true" (
    call :check_dir_exists "%EMBEDDING_DIR%"
    if /I "!DIR_EXISTS!"=="false" (
        if /I "%TILE_METHOD%"=="block2vec" (
            python train_block2vec.py --json_file "%TILE_JSON%" --output_dir "%EMBEDDING_DIR%" --embedding_dim %TILE_DIM% --epochs 200 --batch_size 32
        ) else (
            python train_skipgram.py --json_file "%TILE_JSON%" --output_dir "%EMBEDDING_DIR%" --embedding_dim %TILE_DIM% --epochs 200 --batch_size 32
        )
    )
    python log_timestamp.py --log_file %TIMING_LOG% --event "tile embedding training"
)

REM ===========================================================================
REM Step 1: train our own MLM text encoder (only when USE_MLM is true).
REM If MLM_OUTPUT already exists, training is skipped and the existing
REM model is reused as-is.
REM ===========================================================================
if /I "%USE_MLM%"=="true" (
    call :check_dir_exists "%MLM_OUTPUT%"
    if /I "!DIR_EXISTS!"=="false" (
        python train_mlm.py --epochs %MLM_EPOCHS% --checkpoint_freq %MLM_CHECKPOINT% --save_checkpoints --json %TRAIN_DATA% --val_json %VAL_DATA% --test_json %TEST_DATA% --pkl %TOKENIZER% --output_dir %MLM_OUTPUT% --seed %SEED% %MLM_MAX_SEQ_LENGTH_FLAG%
    )
    python log_timestamp.py --log_file %TIMING_LOG% --event "MLM training"
)

REM ===========================================================================
REM Step 2: diffusion model training.
REM If MODEL_DIR already exists, as to resume.
REM ===========================================================================
set "CAPTION_SCORE_PLOT_FLAG=--plot_validation_caption_score"
if defined CAPTION_SOURCE_KEYS set "CAPTION_SCORE_PLOT_FLAG="

if /I "%UNCONDITIONAL%"=="true" (
    python train_diffusion.py     --save_image_epochs %DIFFUSION_EPOCHS% --augment                    --output_dir "%MODEL_DIR%" --num_epochs %DIFFUSION_EPOCHS% --json %TRAIN_DATA% --val_json %VAL_DATA% --seed %SEED% --game %GAME% %BLOCK_EMBED_FLAG% %BATCH_SIZE_FLAG% %CAPTION_SOURCE_KEYS_ARG%
    if errorlevel 1 (
        echo Error: train_diffusion.py failed.
        exit /b 1
    )
) else (
    if /I "%USE_MLM%"=="true" (
        python train_diffusion.py --save_image_epochs %DIFFUSION_EPOCHS% --augment --text_conditional --output_dir "%MODEL_DIR%" --num_epochs %DIFFUSION_EPOCHS% --json %TRAIN_DATA% --val_json %VAL_DATA% --seed %SEED% --game %GAME% %BLOCK_EMBED_FLAG% %BATCH_SIZE_FLAG% %CAPTIONS_PER_KEY_FLAG% %CAPTION_SOURCE_KEYS_ARG% %DIFF_FLAGS% %DESCRIBE_ABSENCE_FLAG% %CAPTION_SCORE_PLOT_FLAG% --plot_clip_score --pkl %TOKENIZER% --mlm_model_dir %MLM_OUTPUT%
        if errorlevel 1 (
            echo Error: train_diffusion.py failed.
            exit /b 1
        )
    ) else (
        python train_diffusion.py --save_image_epochs %DIFFUSION_EPOCHS% --augment --text_conditional --output_dir "%MODEL_DIR%" --num_epochs %DIFFUSION_EPOCHS% --json %TRAIN_DATA% --val_json %VAL_DATA% --seed %SEED% --game %GAME% %BLOCK_EMBED_FLAG% %BATCH_SIZE_FLAG% %CAPTIONS_PER_KEY_FLAG% %CAPTION_SOURCE_KEYS_ARG% %DIFF_FLAGS% %DESCRIBE_ABSENCE_FLAG% %CAPTION_SCORE_PLOT_FLAG% --plot_clip_score --pretrained_language_model "%MODEL_NAME%" %SPLIT_FLAG%
        if errorlevel 1 (
            echo Error: train_diffusion.py failed.
            exit /b 1
        )
    )
)
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion training"

REM ===========================================================================
REM Step 3: generate samples
REM ===========================================================================
if /I "%UNCONDITIONAL%"=="true" (
    call batch\run_diffusion_multi.bat %MODEL_DIR% regular %GAME%
) else (
    call batch\run_diffusion_multi.bat %MODEL_DIR% %TYPE% %GAME%
)
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion samples"

REM ===========================================================================
REM Step 4: evaluate caption adherence (conditional models only)
REM ===========================================================================
if /I "%UNCONDITIONAL%"=="false" (
    call batch\evaluate_caption_adherence_multi.bat %MODEL_DIR% %TYPE% %DATA% %GAME% %CAPTION_SOURCE_KEYS_ARG%
    python log_timestamp.py --log_file %TIMING_LOG% --event "caption adherence evaluation"
)

REM move the timing log into the trained model's directory
move /Y %TIMING_LOG% %MODEL_DIR%\pipeline_timing.jsonl

exit /b 0

REM ===========================================================================
REM Subroutine: check_dir_exists <dir>
REM Sets DIR_EXISTS to "true" and prints a notice if the given output
REM directory already exists, or "false" otherwise. Callers use DIR_EXISTS
REM to decide whether to skip the corresponding training step.
REM ===========================================================================
:check_dir_exists
set DIR_EXISTS=false
if exist "%~1" (
    echo Notice: Output directory "%~1" already exists. Skipping this training step.
    set DIR_EXISTS=true
)
exit /b 0
