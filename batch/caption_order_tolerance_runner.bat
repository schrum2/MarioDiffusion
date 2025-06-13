cd ..
REM Loop through all directories starting with Mar1and2-conditional and ending in 0
for /D %%D in ("Mar1and2-conditional*0") do (
    set "DIR=%%~nxD"
    
    REM Call a subroutine to preserve variable expansion inside the loop
    call :processDir "%%D" "%%~nxD"
)
pause
exit /b

:processDir
set "FULLDIR=%~1"
set "DIRNAME=%~2"

REM Check if the directory name contains 'absence'
echo %DIRNAME% | findstr /i "absence" >nul
if %errorlevel%==0 (
    set "TYPE=absence"
) else (
    set "TYPE=regular"
)

REM Run the Python script with the correct arguments
python evaluate_caption_order_tolerance.py --json datasets\Mar1and2_LevelsAndCaptions-%TYPE%-test.json --game Mario --save_as_json --model_path %FULLDIR%
exit /b
