REM @echo off
REM Usage: train-conditionali-embeddings.bat <seed> <embedding-len> 
REM <seed> is optional, defaults to 0
REM <embedding-len> is optional, defaults to 16
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set EMBEDDING_DIM=%2
if "%EMBEDDING_DIM%" == "" set EMBEDDING_DIM=16

python create_tile_level_json_data.py --output datasets\SMB1_3x3_tiles.json --tile_size 3 --levels "..\TheVGLC\Super Mario Bros\Processed"
python create_tile_level_json_data.py --output datasets\SMB2_3x3_tiles.json --tile_size 3 --levels "..\TheVGLC\Super Mario Bros 2 (Japan)\Processed"
python combine_data.py datasets\Mar1and2_3x3_tiles.json datasets\SMB1_3x3_tiles.json datasets\SMB2_3x3_tiles.json


set MODEL_PATH="TILE_EMBEDDING%EMBEDDING_DIM%_Mar1and2-conditional-block2vec%SEED%"

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --test_json datasets\Mar1and2_LevelsAndCaptions-regular-test.json --pkl datasets\Mar1and2_Tokenizer-regular.pkl --output_dir TILE_EMBEDDING%EMBEDDING_DIM%_Mar1and2-MLM-regular%SEED% --seed %SEED%
python train_block2vec.py --json_file datasets\Mar1and2_3x3_tiles.json --output_dir "TILE_EMBEDDING%EMBEDDING_DIM%_Mar1and2-block2vec-embeddings%SEED%" --embedding_dim %EMBEDDING_DIM% --epochs 200 --batch_size 32
python train_diffusion.py --augment --text_conditional --output_dir "%MODEL_PATH%" --num_epochs 500 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --mlm_model_dir "TILE_EMBEDDING%EMBEDDING_DIM%_Mar1and2-MLM-regular%SEED%" --block_embedding_model_path "TILE_EMBEDDING%EMBEDDING_DIM%_Mar1and2-block2vec-embeddings%SEED%" --plot_validation_caption_score
python run_diffusion.py --model_path "%MODEL_PATH%" --num_samples 100 --save_as_json --output_dir "TILE_EMBEDDING%EMBEDDING_DIM%_Mar1and2-conditional-block2vec%SEED%-samples"

python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\Mar1and2_RandomTest-regular.json --output_dir samples-from-random-Mar1and2-captions --random_width --width_range_json datasets\Mar1and2_LevelsAndCaptions-regular.json --num_tiles=13
python evaluate_caption_adherence.py --model_path %MODEL_PATH% --save_as_json --json datasets\Mar1and2_RandomTest-regular.json --compare_checkpoints --random_width --width_range_json datasets\Mar1and2_LevelsAndCaptions-regular.json --num_tiles=13
