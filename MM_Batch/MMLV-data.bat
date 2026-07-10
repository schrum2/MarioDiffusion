@echo off

cd ..

set TIMING_LOG=timing_logs\MMLV-data.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "MMLV-data pipeline start"

REM download a ton of levels from MMLV
python megaman\Bulk_Download.py --target 5000
python log_timestamp.py --log_file %TIMING_LOG% --event "bulk download"

REM convert to VGLC ASCII
python megaman\bulk_mmlv_to_vglc.py --output datasets\MMLV_Levels
python log_timestamp.py --log_file %TIMING_LOG% --event "MMLV to VGLC conversion"

REM convert to json
python create_megaman_json_data.py --levels datasets\MMLV_Levels --tileset datasets\MMLV.json --stride_x 16 --stride_y 14 --scan_mode screen_grid --include_moving_ground --output datasets\MMLV_Levels.json
python log_timestamp.py --log_file %TIMING_LOG% --event "build level JSON"

REM assign deterministic captions
python MM_create_ascii_captions.py --dataset datasets\MMLV_Levels.json --tileset datasets\MMLV.json --output datasets\MMLV_LevelsAndCaptions-multi.json --caption-mode keyed --caption-key deterministic_captions
python log_timestamp.py --log_file %TIMING_LOG% --event "deterministic ASCII captions"

REM LLM 1: llama3.1
python llm_ascii_to_caption.py --levels datasets\MMLV_LevelsAndCaptions-multi.json --game mmlv --output datasets\MMLV_LevelsAndCaptions-multi.json --llm ollama --model llama3.1:8b
python log_timestamp.py --log_file %TIMING_LOG% --event "LLM captions (llama3.1:8b)"

REM LLM 2: gemma4 
python llm_ascii_to_caption.py --levels datasets\MMLV_LevelsAndCaptions-multi.json --game mmlv --output datasets\MMLV_LevelsAndCaptions-multi.json --llm ollama --model gemma4:26b
python log_timestamp.py --log_file %TIMING_LOG% --event "LLM captions (gemma4:26b)"

REM split into training/val/test splits 
python split_data.py --json_file datasets\MMLV_LevelsAndCaptions-multi.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MMLV
python log_timestamp.py --log_file %TIMING_LOG% --event "train/validate/test split"
