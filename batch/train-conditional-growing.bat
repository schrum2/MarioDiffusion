REM The idea behind this script is to train a model, but use outputs that don't represent known captions
REM as additional training data, gradually growing the size of the training set over time. It seems to
REM run fine, but it is not clear if it really improves over simply training on the original data. At The
REM very least, there is no noticable improvement on caption adherence across randomized captions.

REM @echo off
REM Usage: train-conditional-growing.bat <seed> <threshold> <max_new_samples> <max_dataset_size>
REM <seed> is optional, defaults to 0
REM <threshold> is the auto_augment_threshold (e.g. 0.8)
REM <max_new_samples> is the auto_augment_max_new_samples (e.g. 10)
REM <max_dataset_size> is the auto_augment_max_dataset_size (e.g. 10000)
cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0

set GAME=Mar1and2
set TYPE=regular

set THRESHOLD=%2
if "%THRESHOLD%"=="" set THRESHOLD=0.9
set MAX_NEW=%3
if "%MAX_NEW%"=="" set MAX_NEW=200
set MAX_SIZE=%4
if "%MAX_SIZE%"=="" set MAX_SIZE=20000

set MLM_OUTPUT=Mario-%GAME%-MLM-%TYPE%%SEED%

set DIFF_OUTPUT=Mario-%GAME%-conditional-%TYPE%-thres%THRESHOLD%new%MAX_NEW%max%MAX_SIZE%-%SEED%

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --test_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-test.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --output_dir %MLM_OUTPUT% --seed %SEED%
python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir "%DIFF_OUTPUT%" --max_iterations 200000 --json datasets\%GAME%_LevelsAndCaptions-%TYPE%-train.json --val_json datasets\%GAME%_LevelsAndCaptions-%TYPE%-validate.json --pkl datasets\%GAME%_Tokenizer-%TYPE%.pkl --mlm_model_dir %MLM_OUTPUT% --plot_validation_caption_score --seed %SEED% --auto_augment --auto_augment_threshold %THRESHOLD% --auto_augment_max_new_samples %MAX_NEW% --auto_augment_max_dataset_size %MAX_SIZE% --num_epochs 1000
call batch\run_diffusion_multi.bat %DIFF_OUTPUT% %TYPE% %GAME% text
call batch\evaluate_caption_adherence_multi.bat %DIFF_OUTPUT% %TYPE% %GAME%
