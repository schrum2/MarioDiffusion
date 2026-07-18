REM @echo off
pushd "%~dp0"
REM Usage: train-conditional-embeddings.bat <seed> <method> <embedding-len>
REM <seed> is optional, defaults to 0
REM <method> is optional, defaults to block2vec. Must be skip or block2vec.
REM <embedding-len> is optional, defaults to 16
cd ..

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

set "SMB1_JSON=Game_Mario/DATA/SMB1_3x3_tiles.json"
set "SMB2_JSON=Game_Mario/DATA/SMB2_3x3_tiles.json"
set "MAR12_JSON=Game_Mario/DATA/Mar1and2_3x3_tiles.json"
set "SMB1_LEVELS=..\TheVGLC\Super Mario Bros\Processed"
set "SMB2_LEVELS=..\TheVGLC\Super Mario Bros 2 (Japan)\Processed"

if not exist "%MAR12_JSON%" (
    python create_tile_level_json_data.py --output "%SMB1_JSON%" --tile_size 3 --levels "%SMB1_LEVELS%"
    python create_tile_level_json_data.py --output "%SMB2_JSON%" --tile_size 3 --levels "%SMB2_LEVELS%"
    python combine_data.py "%MAR12_JSON%" "%SMB1_JSON%" "%SMB2_JSON%"
)

set "EMBEDDING_DIR=Mario-Mar1and2-%METHOD%%EMBEDDING_DIM%-embeddings%SEED%"
set "MODEL_PATH=Mario-Mar1and2-%METHOD%%EMBEDDING_DIM%-conditional%SEED%"
set "MLM_DIR=Mario-Mar1and2-MLM-regular%SEED%"
set "SAMPLES_DIR=Mario-Mar1and2-%METHOD%%EMBEDDING_DIM%-conditional%SEED%-samples"
set "EVAL_ARGS=--model_path %MODEL_PATH% --save_as_json --json Game_Mario/DATA/Mar1and2_RandomTest-regular.json --random_width --width_range_json Game_Mario/DATA/Mar1and2_LevelsAndCaptions-regular.json --num_tiles=13"

if not exist "%MLM_DIR%" (
    python train_mlm.py --epochs 300 --save_checkpoints --json Game_Mario/DATA/Mar1and2_LevelsAndCaptions-regular-train.json --val_json Game_Mario/DATA/Mar1and2_LevelsAndCaptions-regular-validate.json --test_json Game_Mario/DATA/Mar1and2_LevelsAndCaptions-regular-test.json --pkl Game_Mario/DATA/Mar1and2_Tokenizer-regular.pkl --output_dir "%MLM_DIR%" --seed %SEED%
)

if not exist "%EMBEDDING_DIR%" (
    if /I "%METHOD%"=="block2vec" (
        python train_block2vec.py --json_file "%MAR12_JSON%" --output_dir "%EMBEDDING_DIR%" --embedding_dim %EMBEDDING_DIM% --epochs 200 --batch_size 32
    ) else (
        python train_skipgram.py --json_file "%MAR12_JSON%" --output_dir "%EMBEDDING_DIR%" --embedding_dim %EMBEDDING_DIM% --epochs 200 --batch_size 32
    )
)
python train_diffusion.py --augment --text_conditional --output_dir "%MODEL_PATH%" --num_epochs 500 --json Game_Mario/DATA/Mar1and2_LevelsAndCaptions-regular-train.json --val_json Game_Mario/DATA/Mar1and2_LevelsAndCaptions-regular-validate.json --mlm_model_dir "%MLM_DIR%" --block_embedding_model_path "%EMBEDDING_DIR%" --plot_validation_caption_score
python run_diffusion.py --model_path "%MODEL_PATH%" --num_samples 100 --save_as_json --output_dir "%SAMPLES_DIR%"

python evaluate_caption_adherence.py %EVAL_ARGS% --output_dir samples-from-random-Mar1and2-captions
python evaluate_caption_adherence.py %EVAL_ARGS% --compare_checkpoints

popd
exit /b
