@echo off
setlocal enabledelayedexpansion

set "DEFAULT_SAVE_DIR=%~dp0..\datasets\MM_Maker_Levels_Raw"

if not "%~1"=="" (
    set "LEVEL_ID=%~1"
) else (
    set /p LEVEL_ID=Downloading level ID: 
)

if not "%~2"=="" (
    set "SAVE_DIR=%~2"
) else (
    set "SAVE_DIR=%DEFAULT_SAVE_DIR%"
)

if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

python -c "import requests, gzip, os, sys; level_id=sys.argv[1]; save_dir=sys.argv[2]; meta=requests.get('https://api.megamanmaker.com/level/download/' + level_id).json(); open(os.path.join(save_dir, level_id + '.mmlv'), 'w').write(gzip.decompress(requests.get(meta['location']).content).decode()); print('done')" "%LEVEL_ID%" "%SAVE_DIR%"

endlocal