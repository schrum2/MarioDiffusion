
REM @echo off
REM Usage: LR-conditional.bat <seed> <type>
REM <type> should be "regular" or "absence"
REM <seed> is optional, defaults to 0
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set TYPE=%2
if "%TYPE%"=="" set TYPE=regular


set MLM_OUTPUT=LR-MLM-%TYPE%%SEED%
set DIFF_OUTPUT=LR-conditional-%TYPE%%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-unconditional-samples
set GAME=LR

python train_mlm.py --epochs 80000 --save_checkpoints --json datasets\LR_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\LR_LevelsAndCaptions-%TYPE%-validate.json --test_json datasets\LR_LevelsAndCaptions-%TYPE%-test.json --pkl datasets\LR_Tokenizer-%TYPE%.pkl --output_dir %MLM_OUTPUT% --seed %SEED%
python train_diffusion.py --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 25000 --json datasets\LR_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\LR_LevelsAndCaptions-%TYPE%-validate.json --pkl datasets\LR_Tokenizer-%TYPE%.pkl --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% --game %GAME%
python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --text_conditional --save_as_json --output_dir "%UNCOND_OUTPUT%" --game %GAME%
call batch\\evaluate_caption_adherence_multi.bat %DIFF_OUTPUT% %TYPE% %GAME%