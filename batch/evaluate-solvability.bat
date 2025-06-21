cd ..

REM Loop through all directories starting with Mar1and2-conditional in the current directory
for /D %%D in ("Mar1and2-conditional*0") do (
    echo Processing %%D
    python evaluate_solvability.py --num_runs 1 --model_path "%%D"
)
pause

:: Unconditional Models
python evaluate_solvability.py --num_runs 1 --model_path "Mar1and2-unconditional0"

:: FDM Models
python evaluate_solvability.py --num_runs 1 --model_path "Mar1and2-fdm-MiniLM-regular0"
python evaluate_solvability.py --num_runs 1 --model_path "Mar1and2-fdm-MiniLM-absence0"
python evaluate_solvability.py --num_runs 1 --model_path "Mar1and2-fdm-GTE-regular0"
python evaluate_solvability.py --num_runs 1 --model_path "Mar1and2-fdm-GTE-absence0"


:: WGAN Models : Mar1and2-wgan0-samples (evaluate_solvability.py can't handle this, temp script in google drive)
python evaluate_solvability.py --num_runs 1 --model_path "Mar1and2-wgan0"
:: MarioGPT Models : Ask Olivia
