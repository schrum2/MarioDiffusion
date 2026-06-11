@echo off
cd ..
set /p FILE=Upload a file: 

if "%FILE:~-5%"==".mmlv" (
    python -m megaman.mmlv_to_vglc "%FILE%"
) else if "%FILE:~-4%"==".txt" (
    python -m megaman.vglc_to_mmlv "%FILE%"
) else (
    echo Unsupported file type. Please provide a .mmlv or .txt file.
)