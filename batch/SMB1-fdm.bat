cd ..


python train_fdm.py --augment --output_dir "SMB1-fdm-pretrained" --num_epochs 50 --json datasets\\SMB1_LevelsAndCaptions-regular-train.json --val_json datasets\\SMB1_LevelsAndCaptions-regular-validate.json --pretrained_language_model "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" --plot_validation_caption_score
python run_fdm.py --model_path "SMB1-fdm-pretrained\\final-model" --num_samples 100 --output_dir "FDM_output_samples"