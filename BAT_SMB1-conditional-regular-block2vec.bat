python create_tile_level_json_data.py --output "SMB1_3x3_tiles.json" --tile_size 3
python train_block2vec.py --json_file "SMB1_3x3_tiles.json" --output_dir "SMB1-regular-block2vec-embeddings" --embedding_dim 16 --epochs 20 --batch_size 32
python train_diffusion.py ^
    --augment ^
    --text_conditional ^
    --output_dir "SMB1-conditional-block2vec" ^
    --num_epochs 20^
    --json "SMB1_LevelsAndCaptions-regular.json" ^
    --pkl "SMB1_Tokenizer-regular.pkl" ^
    --mlm_model_dir "SMB1-regular-block2vec-embeddings" ^
    --block_embedding_model_path "SMB1-regular-block2vec-embeddings" ^
    --plot_validation_caption_score
python run_diffusion.py ^
    --model_path "SMB1-conditional-block2vec" ^
    --num_samples 100 ^
    --save_as_json ^
    --output_dir "SMB1-conditional-block2vec-samples"
