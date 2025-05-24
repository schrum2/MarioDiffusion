python create_tile_level_json_data.py --output "SMB1_3x3_tiles.json" --tile_size 3
python train_block2vec.py --json_file "SMB1_3x3_tiles.json" --output_dir "SMB1-block2vec-embeddings" --embedding_dim 16 --epochs 100 --batch_size 32
python split_data.py --json SMB1_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python train_diffusion.py --augment --output_dir "SMB1-unconditional-block2vec" --num_epochs 500 --json SMB1_LevelsAndCaptions-regular-train.json --val_json SMB1_LevelsAndCaptions-regular-validate.json --block_embedding_model_path "SMB1-block2vec-embeddings"
python run_diffusion.py --model_path "SMB1-unconditional-block2vec" --num_samples 100 --save_as_json --output_dir "SMB1-unconditional-block2vec-samples"