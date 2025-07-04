cd ..

python create_tile_level_json_data.py --output datasets\\SMB1_3x3_tiles.json --tile_size 3
python train_block2vec.py --json_file datasets\\SMB1_3x3_tiles.json --output_dir "SMB1-block2vec-embeddings" --embedding_dim 16 --epochs 200 --batch_size 32
python train_diffusion.py --augment --output_dir "SMB1-unconditional-block2vec" --num_epochs 500 --json datasets\\SMB1_LevelsAndCaptions-regular-train.json --val_json datasets\\SMB1_LevelsAndCaptions-regular-validate.json --block_embedding_model_path "SMB1-block2vec-embeddings"
python run_diffusion.py --model_path "SMB1-unconditional-block2vec" --num_samples 100 --save_as_json --output_dir "SMB1-unconditional-block2vec-samples"
