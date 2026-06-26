@echo off
REM Usage: MM_unconditional-embeddings.bat <seed> <method> <embedding-len>
REM <seed> is optional, defaults to 0
REM <method> is optional, defaults to block2vec. Must be skip or block2vec.
REM <embedding-len> is optional, defaults to 16

set SEED=%1
if "%SEED%"=="" set SEED=0

set METHOD=%2
set EMBEDDING_DIM=%3
if "%METHOD%"=="" set METHOD=block2vec
if "%EMBEDDING_DIM%"=="" set EMBEDDING_DIM=16

if /I "%METHOD%"=="skip" set "METHOD=skip"
if /I "%METHOD%"=="block2vec" set "METHOD=block2vec"
if /I not "%METHOD%"=="skip" if /I not "%METHOD%"=="block2vec" (
    echo ERROR: Invalid embedding method: %METHOD%
    echo Must be skip or block2vec.
    exit /b 1
)

REM Run MM-data.bat first
cd ..

set "MM_JSON=datasets\MM_3x3_Tiles-simple.json"
set "MM_TILESET=datasets\MM_Simple_Tileset.json"
set "MM_LEVELS=..\TheVGLC\MegaMan\Enhanced"
set "MM_CHAR_MAP=datasets\MM_VGLC_to_Simple.json"
set "EMBEDDING_DIR=MM-simple-%METHOD%%EMBEDDING_DIM%-embeddings%SEED%"
set "MODEL_PATH=MM-simple-unconditional%SEED%-%METHOD%%EMBEDDING_DIM%"
set "SAMPLES_DIR=MM-simple-unconditional%SEED%-%METHOD%%EMBEDDING_DIM%-samples"

if not exist "%MM_JSON%" (
    python create_tile_level_json_data.py --tileset "%MM_TILESET%" --levels "%MM_LEVELS%" --output "%MM_JSON%" --tile_size 3 --char_map "%MM_CHAR_MAP%"
)

if /I "%METHOD%"=="block2vec" (
    python train_block2vec.py --json_file "%MM_JSON%" --output_dir "%EMBEDDING_DIR%" --embedding_dim %EMBEDDING_DIM% --epochs 300
) else (
    python train_skipgram.py --json_file "%MM_JSON%" --output_dir "%EMBEDDING_DIR%" --embedding_dim %EMBEDDING_DIM% --epochs 300
)

python train_diffusion.py --game MM-Simple --augment --block_embedding_model_path "%EMBEDDING_DIR%" --output_dir "%MODEL_PATH%" --num_epochs 500 --json datasets\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\MM_LevelsAndCaptions-simple-regular-validate.json --seed %SEED%
python run_diffusion.py --model_path "%MODEL_PATH%" --num_samples 100 --save_as_json --output_dir "%SAMPLES_DIR%"

