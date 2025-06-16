@echo off

REM Loop through all directories starting with Mar1and2-conditional in the current directory
for /D %%D in ("Mar1and2-conditional*0") do (
    echo Processing %%D
    python evaluate_solvability.py --num_runs 1 --model_path "%%D"
)
pause