@echo off
REM Usage: train-unconditional-embeddings.bat <embedding-len> <seed>
REM <embedding-len> is optional, defaults to 16
REM <seed> is optional, defaults to 0
cd ..

set "EMBEDDING_DIM=%1"
if "%EMBEDDING_DIM%"=="" set "EMBEDDING_DIM=16"

set "SEED=%2"
if "%SEED%"=="" set "SEED=0"

set "SMB1_JSON=datasets\SMB1_3x3_tiles.json"
set "SMB2_JSON=datasets\SMB2_3x3_tiles.json"
set "MAR12_JSON=datasets\Mar1and2_3x3_tiles.json"
set "BLOCK2VEC_DIR=TILE_EMBEDDING%EMBEDDING_DIM%_Mar1and2-block2vec-embeddings%SEED%"
set "MODEL_PATH=TILE_EMBEDDING%EMBEDDING_DIM%_Mar1and2-unconditional-block2vec%SEED%"
set "SAMPLES_DIR=TILE_EMBEDDING%EMBEDDING_DIM%_Mar1and2-unconditional-block2vec%SEED%-samples"

if not exist "%MAR12_JSON%" (
    python create_tile_level_json_data.py --output "%SMB1_JSON%" --tile_size 3 --levels "..\TheVGLC\Super Mario Bros\Processed"
    python create_tile_level_json_data.py --output "%SMB2_JSON%" --tile_size 3 --levels "..\TheVGLC\Super Mario Bros 2 (Japan)\Processed"
    python combine_data.py "%MAR12_JSON%" "%SMB1_JSON%" "%SMB2_JSON%"
)

python train_block2vec.py --json_file "%MAR12_JSON%" --output_dir "%BLOCK2VEC_DIR%" --embedding_dim %EMBEDDING_DIM% --epochs 200 --batch_size 32
python train_diffusion.py --augment --output_dir "%MODEL_PATH%" --num_epochs 500 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --block_embedding_model_path "%BLOCK2VEC_DIR%" --plot_validation_caption_score
python run_diffusion.py --model_path "%MODEL_PATH%" --num_samples 100 --save_as_json --output_dir "%SAMPLES_DIR%"








