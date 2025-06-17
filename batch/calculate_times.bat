@echo off
setlocal enabledelayedexpansion

:: Mar1and2-conditional models (MLM)
python calculate_execution_time.py Mar1and2-conditional-regular 0 9
python calculate_execution_time.py Mar1and2-conditional-absence 0 9
python calculate_execution_time.py Mar1and2-conditional-negative 0 9

:: Mar1and2-conditional-MiniLM models (single sentence)
python calculate_execution_time.py Mar1and2-conditional-MiniLM-regular 0 9
python calculate_execution_time.py Mar1and2-conditional-MiniLM-absence 0 9
python calculate_execution_time.py Mar1and2-conditional-MiniLM-negative 0 9

:: Mar1and2-conditional-MiniLMsplit models (multiple sentences)
python calculate_execution_time.py Mar1and2-conditional-MiniLMsplit-regular 0 4
python calculate_execution_time.py Mar1and2-conditional-MiniLMsplit-absence 0 4
python calculate_execution_time.py Mar1and2-conditional-MiniLMsplit-negative 0 4

:: Mar1and2-conditional-GTE models (single sentence)
python calculate_execution_time.py Mar1and2-conditional-GTE-regular 0 4
python calculate_execution_time.py Mar1and2-conditional-GTE-absence 0 4
python calculate_execution_time.py Mar1and2-conditional-GTE-negative 0 4

:: Mar1and2-conditional-GTEsplit models (multiple sentences)
python calculate_execution_time.py Mar1and2-conditional-GTEsplit-regular 0 0
python calculate_execution_time.py Mar1and2-conditional-GTEsplit-absence 0 0
python calculate_execution_time.py Mar1and2-conditional-GTEsplit-negative 0 0

:: Mar1and2-unconditional models 
python calculate_execution_time.py Mar1and2-unconditional 0 29



REM TODO: FDM, WGAN

echo All execution time calculations complete!
pause
