python create_tile_level_json_data.py --json SMB1_LevelsAndCaptions-regular.json --output_dir block2vec_test_tiles --split
python train_block2vec.py --epochs 300 --save_checkpoints --json block2vec_test_tiles.json --pkl SMB1_Tokenizer-regular.pkl --output_dir SMB1-block2vec-regular --split --embedding_dim 32 --batch_size 256
python split_data.py --json SMB1_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python train_diffusion.py --augment --text_conditional --output_dir "SMB1-conditional-regular-block2vec" --num_epochs 100 --json SMB1_LevelsAndCaptions-regular-train.json --val_json SMB1_LevelsAndCaptions-regular-validate.json --pkl SMB1_Tokenizer-regular.pkl --mlm_model_dir SMB1-block2vec-regular --plot_validation_caption_score
