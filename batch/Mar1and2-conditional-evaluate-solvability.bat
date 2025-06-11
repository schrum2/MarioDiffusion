@echo off

REM Loop through all directories starting with Mar1and2-conditional in the current directory
for /D %%D in ("Mar1and2-conditional*") do (
    echo Processing %%D
    python evaluate_solvability.py --model_path "%%D"
)
pause