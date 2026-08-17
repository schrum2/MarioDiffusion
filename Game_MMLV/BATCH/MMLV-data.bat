@echo off
cd ..
cd ..

set TIMING_LOG=timing_logs\MMLV-data.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "MMLV-data pipeline start"

set LVL_TARGET=%~1
if "%LVL_TARGET%"=="" set LVL_TARGET=5000

REM download a ton of levels from MMLV
python Game_MMLV\bulk_mmlv_download.py --target %LVL_TARGET%
python log_timestamp.py --log_file %TIMING_LOG% --event "bulk download"

REM convert to VGLC ASCII
python Game_MMLV\bulk_mmlv_to_vglc.py --output Game_MMLV\MMLV_Levels
python log_timestamp.py --log_file %TIMING_LOG% --event "MMLV to VGLC conversion"

REM convert to json
python create_megaman_json_data.py --levels Game_MMLV\MMLV_Levels --tileset Game_MMLV\MMLV.json --stride_x 16 --stride_y 14 --scan_mode screen_grid --include_moving_ground --output Game_MMLV\DATA\MMLV_Levels.json --max_enemies 8 --min_content_pct 7
python log_timestamp.py --log_file %TIMING_LOG% --event "build level JSON"

REM assign deterministic captions
python MM_create_ascii_captions.py --dataset Game_MMLV\DATA\MMLV_Levels.json --tileset Game_MMLV\MMLV.json --output Game_MMLV\DATA\MMLV_LevelsAndCaptions-regular.json --caption-mode keyed --caption-key deterministic_captions
python log_timestamp.py --log_file %TIMING_LOG% --event "deterministic ASCII captions"

