REM @echo off
pushd "%~dp0"
REM Usage: MM_conditional-embeddings.bat [seed] [simple|full] [block2vec/skip] [embedding-len] [window-size]
REM <variant> is optional, defaults to simple.
REM <method> is optional, defaults to block2vec. Must be skip or block2vec.
REM <embedding-len> is optional, defaults to 16.
REM <window-size> is optional, defaults to 3 and must be an odd number.

set SEED=%1
if "%SEED%"=="" set SEED=0

set VARIANT=%2
if "%VARIANT%"=="" set VARIANT=simple

set METHOD=%3
set EMBEDDING_DIM=%4
set WINDOW_SIZE=%5
if "%METHOD%"=="" set METHOD=block2vec
if "%EMBEDDING_DIM%"=="" set EMBEDDING_DIM=16
if "%WINDOW_SIZE%"=="" set WINDOW_SIZE=3

if /I "%VARIANT%"=="simple" (
    set "DATASET_INFIX=simple"
    set "GAME=MM-Simple"
    set "MM_TILESET=datasets\MM-simple-tileset.json"
    set "MM_LEVELS=..\TheVGLC\MegaMan\Enhanced"
    set "MM_CHAR_MAP=datasets\MM-VGLC-to-simple.json"
    set "NUM_TILES=13"
) else if /I "%VARIANT%"=="full" (
    set "DATASET_INFIX=full"
    set "GAME=MM-Full"
    set "MM_TILESET=datasets\MM.json"
    set "MM_LEVELS=..\TheVGLC\MegaMan\Enhanced"
    set "MM_CHAR_MAP="
    set "NUM_TILES=41"
) else (
    echo ERROR: Invalid dataset variant: %VARIANT%
    echo Must be simple or full.
    exit /b 1
)

if /I "%METHOD%"=="skip" set "METHOD=skip"
if /I "%METHOD%"=="block2vec" set "METHOD=block2vec"
if /I not "%METHOD%"=="skip" if /I not "%METHOD%"=="block2vec" (
    echo ERROR: Invalid embedding method: %METHOD%
    echo Must be skip or block2vec.
    exit /b 1
)

set /a "WINDOW_SIZE=%WINDOW_SIZE%"
set /a "WINDOW_MOD=WINDOW_SIZE %% 2"
if "%WINDOW_MOD%"=="0" (
    echo ERROR: Window size must be an odd number.
    exit /b 1
)
if %WINDOW_SIZE% LSS 1 (
    echo ERROR: Window size must be greater than 0.
    exit /b 1
)

REM Run MM-data.bat first
cd ..

set "MM_JSON=datasets\MM_%WINDOW_SIZE%x%WINDOW_SIZE%_Tiles-%DATASET_INFIX%.json"
set "EMBEDDING_DIR=MM-%DATASET_INFIX%-%METHOD%%EMBEDDING_DIM%-w%WINDOW_SIZE%-embeddings%SEED%"
set "MODEL_PATH=MM-%DATASET_INFIX%-%METHOD%%EMBEDDING_DIM%-w%WINDOW_SIZE%-conditional-regular%SEED%"
set "MLM_DIR=MM-%DATASET_INFIX%-MLM-regular%SEED%"
set "SAMPLES_DIR=MM-%DATASET_INFIX%-%METHOD%%EMBEDDING_DIM%-w%WINDOW_SIZE%-conditional-regular%SEED%-samples"
set "MLM_JSON=datasets\MM-%DATASET_INFIX%_LevelsAndCaptions-regular.json"
set "TOKENIZER_PKL=datasets\MM-%DATASET_INFIX%_Tokenizer-regular.pkl"
set "DIFF_TRAIN_JSON=datasets\MM-%DATASET_INFIX%_LevelsAndCaptions-regular-train.json"
set "DIFF_VAL_JSON=datasets\MM-%DATASET_INFIX%_LevelsAndCaptions-regular-validate.json"
set "RANDOM_TEST_JSON=datasets\MM-%DATASET_INFIX%_RandomTest-regular.json"
set "WIDTH_RANGE_JSON=datasets\MM-%DATASET_INFIX%_LevelsAndCaptions-regular.json"
set "EVAL_ARGS=--model_path "%MODEL_PATH%" --save_as_json --json "%RANDOM_TEST_JSON%" --random_width --width_range_json "%WIDTH_RANGE_JSON%" --num_tiles=%NUM_TILES% --game %GAME%"

REM Per-execution timing log: staged under timing_logs\ during the run, then moved
REM into the trained diffusion model's directory at the end.
set "TIMING_LOG=timing_logs\%MODEL_PATH%.jsonl"
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file "%TIMING_LOG%" --status start --event "MM_conditional-embeddings pipeline start"

if not exist "%MLM_DIR%" (
    python train_mlm.py --epochs 300 --save_checkpoints --json "%MLM_JSON%" --pkl "%TOKENIZER_PKL%" --output_dir "%MLM_DIR%" --seed %SEED%
) else (
    echo Using existing MLM model: %MLM_DIR%
)
python log_timestamp.py --log_file "%TIMING_LOG%" --event "MLM model ready"

if not exist "%MM_JSON%" (
    if /I "%VARIANT%"=="simple" (
        python create_tile_level_json_data.py --tileset "%MM_TILESET%" --levels "%MM_LEVELS%" --output "%MM_JSON%" --tile_size %WINDOW_SIZE% --char_map "%MM_CHAR_MAP%"
    ) else (
        python create_tile_level_json_data.py --tileset "%MM_TILESET%" --levels "%MM_LEVELS%" --output "%MM_JSON%" --tile_size %WINDOW_SIZE%
    )
)
python log_timestamp.py --log_file "%TIMING_LOG%" --event "tile windows dataset ready"

if not exist "%EMBEDDING_DIR%" (
    if /I "%METHOD%"=="block2vec" (
        python train_block2vec.py --json_file "%MM_JSON%" --output_dir "%EMBEDDING_DIR%" --embedding_dim %EMBEDDING_DIM% --epochs 300
    ) else (
        python train_skipgram.py --json_file "%MM_JSON%" --output_dir "%EMBEDDING_DIR%" --embedding_dim %EMBEDDING_DIM% --epochs 300
    )
)
python log_timestamp.py --log_file "%TIMING_LOG%" --event "embedding training"

python train_diffusion.py --text_conditional --mlm_model_dir "%MLM_DIR%" --game %GAME% --augment --block_embedding_model_path "%EMBEDDING_DIR%" --output_dir "%MODEL_PATH%" --num_epochs 500 --json "%DIFF_TRAIN_JSON%" --val_json "%DIFF_VAL_JSON%" --seed %SEED%
python log_timestamp.py --log_file "%TIMING_LOG%" --event "diffusion training"
python run_diffusion.py --model_path "%MODEL_PATH%" --num_samples 100 --save_as_json --output_dir "%SAMPLES_DIR%" --game %GAME%
python log_timestamp.py --log_file "%TIMING_LOG%" --event "sampling"

python evaluate_caption_adherence.py %EVAL_ARGS% --output_dir samples-from-random-MM-%DATASET_INFIX%-captions
python evaluate_caption_adherence.py %EVAL_ARGS% --compare_checkpoints
python log_timestamp.py --log_file "%TIMING_LOG%" --event "caption adherence evaluation"

REM move the timing log into the trained model's directory
move /Y "%TIMING_LOG%" "%MODEL_PATH%\pipeline_timing.jsonl"

popd
exit /b