REM @echo off

REM EXPERIMENTAL: trains a text encoder + conditional diffusion model on the large mixed-width
REM Mario dataset (widths 16, 32, 64, and 128 combined). Run batch\Mar1and2Mixed-data.bat first
REM to build that dataset. It is much larger than the single-width sets, so training is slow and
REM memory-hungry; batch_size is reduced to 16 so the 128-wide scenes fit in memory.
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0



set DATA=Mar1and2_16-32-64-128
set LABEL=Mar1and2Mixed

set MLM_OUTPUT=%LABEL%-MLM-regular%SEED%
set DIFF_OUTPUT=%LABEL%-conditional-regular%SEED%


python train_mlm.py --epochs 300 --save_checkpoints --json datasets\%DATA%_LevelsAndCaptions-regular-train.json --val_json datasets\%DATA%_LevelsAndCaptions-regular-validate.json --test_json datasets\%DATA%_LevelsAndCaptions-regular-test.json --pkl datasets\%LABEL%_Tokenizer-regular.pkl --output_dir %MLM_OUTPUT% --seed %SEED%
python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --num_epochs 300 --json datasets\%DATA%_LevelsAndCaptions-regular-train.json --val_json datasets\%DATA%_LevelsAndCaptions-regular-validate.json --pkl datasets\%LABEL%_Tokenizer-regular.pkl --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% --batch_size 16 
call batch\run_diffusion_multi.bat %DIFF_OUTPUT% regular %DATA% text
