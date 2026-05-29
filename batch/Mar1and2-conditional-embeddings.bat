pushd "%~dp0.."

set EMBEDDING_DIM=%1
if "%EMBEDDING_DIM%" == "" set EMBEDDING_DIM=16


python create_tile_level_json_data.py --output datasets\SMB1_3x3_tiles.json --tile_size 3
python create_tile_level_json_data.py --output datasets\SMB2_3x3_tiles.json --tile_size 3 --levels "..\TheVGLC\Super Mario Bros 2 (Japan)\Processed"
python combine_data.py datasets\Mar1and2_3x3_tiles.json datasets\SMB1_3x3_tiles.json datasets\SMB2_3x3_tiles.json



python train_mlm.py --epochs 300 --save_checkpoints --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --test_json datasets\Mar1and2_LevelsAndCaptions-regular-test.json --pkl datasets\Mar1and2_Tokenizer-regular.pkl --output_dir Mar1and2-MLM-regular0 --seed 0
python train_block2vec.py --json_file datasets\Mar1and2_3x3_tiles.json --output_dir "Mar1and2-block2vec-embeddings" --embedding_dim %EMBEDDING_DIM% --epochs 200 --batch_size 32
python train_diffusion.py --augment --text_conditional --output_dir "Mar1and2-conditional-block2vec" --num_epochs 500 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --mlm_model_dir "Mar1and2-MLM-regular0" --block_embedding_model_path "Mar1and2-block2vec-embeddings"
python run_diffusion.py --model_path "Mar1and2-conditional-block2vec" --num_samples 100 --save_as_json --output_dir "Mar1and2-conditional-block2vec-samples"








