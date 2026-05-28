@echo off
REM MM-conditional.bat
REM Trains a text-conditional diffusion model for Mega Man (MM-Simple).
REM Usage: MM-conditional.bat [seed]
REM [seed] is optional, defaults to 0
 
set SEED=%1
if "%SEED%"=="" set SEED=0
 
cd ..
 
:: Step 1: Build datasets, tokenizers, splits, and random test captions
call MM_Batch\MM-data.bat
 
:: Step 2: Train text encoder (MLM)
python train_mlm.py --json datasets\\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\\MM_LevelsAndCaptions-simple-regular-validate.json --test_json datasets\\MM_LevelsAndCaptions-simple-regular-test.json --output_dir MM-MLM-simple-regular --save_checkpoints --pkl datasets\\MM_Tokenizer-simple-regular.pkl --seed %SEED%
 
:: Step 3: Train text-conditional diffusion model
python train_diffusion.py --pkl datasets\\MM_Tokenizer-simple-regular.pkl --json datasets\\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\\MM_LevelsAndCaptions-simple-regular-validate.json --augment --mlm_model_dir MM-MLM-simple-regular --text_conditional --output_dir MM_conditional_simple_regular%SEED% --seed %SEED% --game MM-Simple --plot_validation_caption_score
