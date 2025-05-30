REM @echo off
REM Usage: batch_runner.bat <job_batch_file> <seed_start> <seed_end> [extra_params...]
REM Example: batch_runner.bat SMB1-conditional-MiniLM.bat 0 4 regular split

setlocal enabledelayedexpansion

set JOB_BATCH=%1
set SEED_START=%2
set SEED_END=%3

if "%JOB_BATCH%"=="" (
    echo ERROR: Must provide a job batch file as first argument.
    exit /b 1
)

if "%SEED_START%"=="" set SEED_START=0
if "%SEED_END%"=="" set SEED_END=%SEED_START%

REM Build the extra parameters string (parameters after seed_end)
set "EXTRA_PARAMS="
shift & shift & shift
:collect_params
if "%~1" neq "" (
    set "EXTRA_PARAMS=%EXTRA_PARAMS% %~1"
    shift
    goto collect_params
)

for /L %%S in (%SEED_START%,1,%SEED_END%) do (
    call "%JOB_BATCH%" %%S%EXTRA_PARAMS%
    cd batch
)