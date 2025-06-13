cd ..
set run = 0
set max_run = 5
:loop_start
if !run! GEQ %max_run% goto end

REM Loop through all directories starting with Mar1and2-conditional and ending in 0
for /D %%D in ("Mar1and2-conditional*0") do (
    set "DIR=%%~nxD"
    call :processDir "%%D" "%%~nxD"
)

set /a run+=1
goto loop_start

:end
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
