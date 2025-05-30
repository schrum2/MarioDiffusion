
REM @echo off
REM Usage: LR-conditional.bat <type> <seed>
REM <type> should be "regular" or "absence"
REM <seed> is optional, defaults to 0
cd ..

set TYPE=%1
if "%TYPE%"=="" set TYPE=regular

set SEED=%2
if "%SEED%"=="" set SEED=0

set MLM_OUTPUT=LR-MLM-%TYPE%%SEED%
set DIFF_OUTPUT=LR-conditional-%TYPE%%SEED%
set UNCOND_OUTPUT=%DIFF_OUTPUT%-unconditional-samples

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\LR_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\LR_LevelsAndCaptions-%TYPE%-validate.json --test_json datasets\LR_LevelsAndCaptions-%TYPE%-test.json --pkl datasets\LR_Tokenizer-%TYPE%.pkl --output_dir %MLM_OUTPUT% --seed %SEED%
python train_diffusion.py --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 500 --json datasets\LR_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\LR_LevelsAndCaptions-%TYPE%-validate.json --pkl datasets\LR_Tokenizer-%TYPE%.pkl --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED%
python run_diffusion.py --model_path %DIFF_OUTPUT% --num_samples 100 --text_conditional --save_as_json --output_dir "%UNCOND_OUTPUT%"
call batch\\LR_evaluate_caption_adherence_multi.bat %DIFF_OUTPUT% %TYPE%