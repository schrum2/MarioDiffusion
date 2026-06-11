@echo off
cd ..
set MMM=%LOCALAPPDATA%\MegaMaker\Levels
set /p FILE=Upload a file: 

if "%FILE:~-5%"==".mmlv" (
    python -m megaman.mmlv_to_vglc "%FILE%" levels\output.txt
    echo Saved to levels\output.txt
) else if "%FILE:~-4%"==".txt" (
    python -m megaman.vglc_to_mmlv "%FILE%" "%MMM%\output.mmlv"
    echo Saved to %MMM%\output.mmlv
) else (
    echo Unsupported file type. Please provide a .mmlv or .txt file.
)