
set EMBEDDING_DIM=%1
if "%EMBEDDING_DIM%" == "" set EMBEDDING_DIM=16

REM Run MM-data.bat first

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-simple-regular.json --pkl datasets\MM_Tokenizer-simple-regular.pkl --output_dir MM-MLM-simple0 --seed 0

python create_tile_level_json_data.py --tileset datasets\MM_Simple_Tileset.json --levels ..\TheVGLC\MegaMan\Enhanced --output datasets\MM_3x3_Tiles-simple.json --tile_size 3 --char_map datasets\MM_VGLC_to_Simple.json

python train_block2vec.py --json_file datasets\MM_3x3_Tiles-simple.json --output_dir MM-simple-block2vec%EMBEDDING_DIM%-embeddings --embedding_dim %EMBEDDING_DIM% --epochs 300

python train_diffusion.py --text_conditional --mlm_model_dir MM-MLM-simple0 --game MM-Simple --augment --block_embedding_model_path MM-simple-block2vec%EMBEDDING_DIM%-embeddings --output_dir MM-simple-conditional0-block2vec%EMBEDDING_DIM% --num_epochs 500 --json datasets\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\MM_LevelsAndCaptions-simple-regular-validate.json --seed 0 
