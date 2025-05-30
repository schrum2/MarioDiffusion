REM @echo off
REM Usage: batch_runner.bat <seed_start> <seed_end> <job_batch_file>
REM Example: batch_runner.bat 0 9 SMB1-conditional-regular-job.bat

setlocal enabledelayedexpansion

set JOB_BATCH=%1
set TYPE=%2

set SEED_START=%3
if "%SEED_START%"=="" set SEED_START=0

set SEED_END=%4
if "%SEED_END%"=="" set SEED_END=%SEED_START%

if "%JOB_BATCH%"=="" (
    echo ERROR: Must provide a job batch file as first argument.
    exit /b 1
)

if "%TYPE%"=="" (
    echo ERROR: Must provide a type of regular or absence as second argument.
    exit /b 1
)

for /L %%S in (%SEED_START%,1,%SEED_END%) do (
    call "%JOB_BATCH%" %TYPE% %%S
	cd batch
)