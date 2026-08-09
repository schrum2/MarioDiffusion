@echo off
REM Usage: MM_unconditional_big.bat [seed]
REM [seed] is optional, defaults to 0. Output goes to megaman\MM-unconditional-big-<seed>,
REM so repeated runs with different seeds don't overwrite each other.

cd ..

set SEED=%1
if "%SEED%"=="" set SEED=0
set MODEL_DIR=megaman\MM-unconditional-big-%SEED%

REM Per-execution timing log: each step appends a timestamped record. The log is
REM staged under timing_logs\ during the run then moved into the trained model's directory at the end.
set TIMING_LOG=timing_logs\MM-unconditional-big-%SEED%.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "MM_unconditional_big pipeline start"

python Game_MMLV\bulk_mmlv_download.py --target 1000
python log_timestamp.py --log_file %TIMING_LOG% --event "MMLV download"

python Game_MMLV\bulk_mmlv_to_vglc.py --output megaman\vglc_out
python log_timestamp.py --log_file %TIMING_LOG% --event "MMLV to VGLC conversion"

python create_megaman_json_data.py --scan_mode screen_grid --target_height 32 --target_width 32 --stride_x 16 --stride_y 14 --levels megaman\vglc_out --tileset Game_MMLV\MMLV.json --output megaman\32x32_MMLV_Levels.json
python log_timestamp.py --log_file %TIMING_LOG% --event "scene sampling to JSON"

REM this part is just so we populate the "caption" field in each json entry, which is required in train_diffusion even for unconditional models
python MM_create_ascii_captions.py --dataset megaman\32x32_MMLV_Levels.json --tileset Game_MMLV\MMLV.json --output megaman\32x32_MMLV_LevelsAndCaptions.json
python log_timestamp.py --log_file %TIMING_LOG% --event "ASCII captioning"

python split_data.py --json_file megaman\32x32_MMLV_LevelsAndCaptions.json --game mmlv --train_pct .9 --val_pct .05 --test_pct .05
python log_timestamp.py --log_file %TIMING_LOG% --event "train/validate/test split"

python train_diffusion.py  --mixed_precision bf16 --model_dim 192 --dim_mults 1 2 4 4 --attention_head_dim 8 --down_block_types "DownBlock2D" "AttnDownBlock2D" "AttnDownBlock2D" "AttnDownBlock2D" --up_block_types "AttnUpBlock2D" "AttnUpBlock2D" "AttnUpBlock2D" "UpBlock2D" --augment --batch_size 8 --game MMLV --json megaman\32x32_MMLV_LevelsAndCaptions-train.json --val_json megaman\32x32_MMLV_LevelsAndCaptions-validate.json --output_dir %MODEL_DIR% --num_epochs 300 --save_image_epochs 10 --max_grad_norm 1.0 --seed %SEED%
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion training"

REM move the timing log into the trained model's directory
move /Y %TIMING_LOG% %MODEL_DIR%\pipeline_timing.jsonl
