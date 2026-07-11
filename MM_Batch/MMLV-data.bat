@echo off

cd ..

set TIMING_LOG=timing_logs\MMLV-data.jsonl
if exist "%TIMING_LOG%" del "%TIMING_LOG%"
python log_timestamp.py --log_file %TIMING_LOG% --status start --event "MMLV-data pipeline start"

REM parse args in any order: a number sets the level target,
REM the literal "no_traversability_filter" disables the A* filter
set LVL_TARGET=
set TRAV_FILTER=
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="no_traversability_filter" (
    set TRAV_FILTER=--no_traversable_filter
) else (
    set LVL_TARGET=%~1
)
shift
goto parse_args
:args_done

if "%LVL_TARGET%"=="" set LVL_TARGET=5000

REM download a ton of levels from MMLV
python megaman\Bulk_Download.py --target %LVL_TARGET%
python log_timestamp.py --log_file %TIMING_LOG% --event "bulk download"

REM convert to VGLC ASCII
python megaman\bulk_mmlv_to_vglc.py --output datasets\MMLV_Levels
python log_timestamp.py --log_file %TIMING_LOG% --event "MMLV to VGLC conversion"

REM convert to json
python create_megaman_json_data.py --levels datasets\MMLV_Levels --tileset datasets\MMLV.json --stride_x 16 --stride_y 14 --scan_mode screen_grid --include_moving_ground --output datasets\MMLV_Levels.json %TRAV_FILTER%
python log_timestamp.py --log_file %TIMING_LOG% --event "build level JSON"

REM assign deterministic captions
python MM_create_ascii_captions.py --dataset datasets\MMLV_Levels.json --tileset datasets\MMLV.json --output datasets\MMLV_LevelsAndCaptions-multi.json --caption-mode keyed --caption-key deterministic_captions
python log_timestamp.py --log_file %TIMING_LOG% --event "deterministic ASCII captions"

