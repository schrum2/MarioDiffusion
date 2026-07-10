@echo off

cd ..

set SIZE=%1
if "%SIZE%"=="" (
	set WIDTH=16
	set HEIGHT=14
) else (
	set WIDTH=%SIZE%
	set HEIGHT=%SIZE%
)

set MODEL_DIR=MM-simple%WIDTH%-filtered-conditional

REM Per-execution timing log: staged under timing_logs\ during the run, then moved
REM into the trained diffusion model's directory at the end.
set TIMING_LOG=timing_logs\%MODEL_DIR%.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "MM_conditional-filtered pipeline start"

python create_megaman_json_data.py --output datasets\MM_Levels-simple%WIDTH%-filtered.json --group_encodings --target_width %WIDTH% --target_height %HEIGHT%
python log_timestamp.py --log_file %TIMING_LOG% --event "scene sampling to JSON"

python MM_create_ascii_captions.py --dataset datasets\MM_Levels-simple%WIDTH%-filtered.json --tileset datasets\MM-simple-tileset.json --output datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular.json
python log_timestamp.py --log_file %TIMING_LOG% --event "ASCII captioning"

python tokenizer.py save --json_file datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular.json --pkl_file datasets\MM_Tokenizer-simple%WIDTH%-filtered-regular.pkl
python log_timestamp.py --log_file %TIMING_LOG% --event "build tokenizer"

python create_random_test_captions.py --save_file datasets\MM_RandomTest_simple%WIDTH%-filtered-regular.json --json datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular.json --seed 0 --game MM-Simple
python log_timestamp.py --log_file %TIMING_LOG% --event "create random test captions"

python split_data.py --json_file datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular.json --train_pct .9 --val_pct .05 --test_pct .05 --seed 0 --game mm-simple
python log_timestamp.py --log_file %TIMING_LOG% --event "train/validate/test split"

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular.json --pkl datasets\MM_Tokenizer-simple%WIDTH%-filtered-regular.pkl --output_dir MM-MLM-simple%WIDTH%-filtered-regular --seed 0
python log_timestamp.py --log_file %TIMING_LOG% --event "MLM training"

python train_diffusion.py --text_conditional --mlm_model_dir MM-MLM-simple%WIDTH%-filtered-regular --game MM-Simple --augment --output_dir %MODEL_DIR% --num_epochs 500 --json datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular-train.json --val_json datasets\MM_LevelsAndCaptions-simple%WIDTH%-filtered-regular-validate.json --seed 0
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion training"

REM move the timing log into the trained model's directory
move /Y %TIMING_LOG% "%MODEL_DIR%\pipeline_timing.jsonl"
