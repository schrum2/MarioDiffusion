@echo off


set width=%1
set height=%1
if "%width%"=="" set width=16
if "%height%"=="" set height=14

set width_suffix=%width%
if "%width%"=="" set width_suffix=

cd ..

set MODEL_DIR=MM-simple%width_suffix%-unconditional0

REM Per-execution timing log: staged under timing_logs\ during the run, then moved
REM into the trained diffusion model's directory at the end.
set TIMING_LOG=timing_logs\%MODEL_DIR%.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "MM_unconditional pipeline start"

python create_megaman_json_data.py --output datasets\MM-simple_Levels%width_suffix%.json --target_height %height% --target_width %width% --group_encodings
python log_timestamp.py --log_file %TIMING_LOG% --event "raw level sampling to JSON"

python MM_create_ascii_captions.py --dataset datasets\MM-simple_Levels%width_suffix%.json --tileset datasets\MM-simple-tileset.json --output datasets\MM-simple_LevelsAndCaptions%width_suffix%-regular.json
python log_timestamp.py --log_file %TIMING_LOG% --event "ASCII captioning"

python tokenizer.py save --json_file datasets\MM-simple_LevelsAndCaptions%width_suffix%-regular.json --pkl_file datasets\MM_Tokenizer-simple%width_suffix%-regular.pkl
python log_timestamp.py --log_file %TIMING_LOG% --event "build tokenizer"

python create_random_test_captions.py --save_file datasets\MM-simple_RandomTest_%width_suffix%-regular.json --json datasets\MM-simple_LevelsAndCaptions%width_suffix%-regular.json --seed 0 --game MM-Simple
python log_timestamp.py --log_file %TIMING_LOG% --event "create random test captions"

python split_data.py --json_file datasets\MM-simple_LevelsAndCaptions%width_suffix%-regular.json --train_pct .9 --val_pct .05 --test_pct .05 --seed 0 --game mm-simple
python log_timestamp.py --log_file %TIMING_LOG% --event "train/validate/test split"

python train_diffusion.py --game MM-Simple --augment --output_dir %MODEL_DIR% --num_epochs 500 --json datasets\MM-simple_LevelsAndCaptions%width_suffix%-regular-train.json --val_json datasets\MM-simple_LevelsAndCaptions%width_suffix%-regular-validate.json --seed 0
python log_timestamp.py --log_file %TIMING_LOG% --event "diffusion training"

REM move the timing log into the trained model's directory
move /Y %TIMING_LOG% "%MODEL_DIR%\pipeline_timing.jsonl"