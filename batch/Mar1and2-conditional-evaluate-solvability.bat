@echo off

REM Loop through all directories starting with Mar1and2-conditional in the current directory
for /D %%D in ("Mar1and2-conditional*0") do (
    REM Skip Mar1and2-conditional-regular0
    if /I not "%%~nxD"=="Mar1and2-conditional-regular0" (
        echo Processing %%D
        python evaluate_solvability.py --model_path "%%D"
    ) else (
        echo Skipping Mar1and2-conditional-regular0 %%D
    )
)
pause