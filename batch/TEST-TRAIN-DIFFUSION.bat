@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM run_tests.bat
REM
REM Runs a battery of train-diffusion.bat invocations covering the distinct
REM execution pathways (unconditional / MLM-conditional / pretrained-
REM conditional, each crossed with no-tile-embed / block2vec / skip, plus
REM both split modes for the pretrained-conditional pathway).
REM
REM Fixed across all tests: seed=99, data=Mar1and2, game=Mario.
REM
REM For each test:
REM   - stdout/stderr is captured to test_logs\<test_name>.log
REM   - a PASS/FAIL line is appended to the report (test_report.txt)
REM   - FAIL is triggered either by a non-zero exit code from
REM     train-diffusion.bat, or by finding an error marker (Traceback,
REM     "Error:", CUDA OOM, etc.) in that test's log, since train-
REM     diffusion.bat does not currently propagate failures from the
REM     underlying python calls into its own exit code.
REM
REM This script assumes it lives in the same directory as train-diffusion.bat
REM (i.e. the batch\ folder).
REM ============================================================================

pushd "%~dp0"

set SEED=99
set DATA=Mar1and2
set GAME=Mario

set LOGDIR=test_logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

set REPORT=test_report.txt
echo Test run started: %DATE% %TIME% > "%REPORT%"
echo. >> "%REPORT%"

set PASS_COUNT=0
set FAIL_COUNT=0

REM ===========================================================================
REM Group A: unconditional pathway (type=none). model/split are ignored by
REM train-diffusion.bat in this pathway, so fixed placeholders are used.
REM ===========================================================================
for %%E in (none block2vec skip) do (
    call :run_test uncond none none single %%E
)

REM ===========================================================================
REM Group B: MLM-conditional pathway. split is ignored in this pathway, so a
REM fixed placeholder is used.
REM ===========================================================================
for %%E in (none block2vec skip) do (
    call :run_test mlm regular MLM single %%E
)

REM ===========================================================================
REM Group C: pretrained-conditional pathway (MiniLM), both split modes.
REM ===========================================================================
for %%S in (single multiple) do (
    for %%E in (none block2vec skip) do (
        call :run_test pretrained regular MiniLM %%S %%E
    )
)

echo. >> "%REPORT%"
echo ============================================================ >> "%REPORT%"
echo Summary: !PASS_COUNT! passed, !FAIL_COUNT! failed >> "%REPORT%"
echo Test run finished: %DATE% %TIME% >> "%REPORT%"

echo.
echo Done. !PASS_COUNT! passed, !FAIL_COUNT! failed.
echo See %REPORT% and %LOGDIR%\ for details.

popd
exit /b 0

REM ===========================================================================
REM Subroutine: run_test <label> <type> <model> <split> <tile_method>
REM
REM Always called with these 5 positional args, in this order. For the
REM unconditional pathway (type=none), <model> and <split> are meaningless
REM placeholders since train-diffusion.bat ignores them in that case.
REM ===========================================================================
:run_test
set TEST_LABEL=%1
set T_TYPE=%2
set T_MODEL=%3
set T_SPLIT=%4
set T_TILE=%5

REM tile_embed_dim is fixed at 8 whenever a tile method is actually used;
REM harmless (ignored) to pass it even when T_TILE is "none".
set T_DIM=8

set TEST_NAME=%TEST_LABEL%_%T_TYPE%_%T_MODEL%_%T_SPLIT%_%T_TILE%
set LOGFILE=%LOGDIR%\%TEST_NAME%.log

echo Running test: %TEST_NAME% ...
echo [%TEST_NAME%] > "%LOGFILE%"
echo Command: train-diffusion.bat %SEED% %DATA% %T_TYPE% %GAME% %T_MODEL% %T_SPLIT% %T_TILE% %T_DIM% >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM Run in an isolated cmd instance so train-diffusion.bat's own "cd .." and
REM "exit /b" calls don't affect this runner's working directory or flow.
cmd /c "train-diffusion.bat %SEED% %DATA% %T_TYPE% %GAME% %T_MODEL% %T_SPLIT% %T_TILE% %T_DIM%" >> "%LOGFILE%" 2>&1
set EXITCODE=!ERRORLEVEL!

set FOUND_ERROR_MARKER=false
findstr /I /C:"Traceback" /C:"Error:" /C:"CUDA out of memory" /C:"CUDA error" "%LOGFILE%" >nul 2>&1
if !ERRORLEVEL! EQU 0 set FOUND_ERROR_MARKER=true

REM Combine both failure conditions into a single flag first - chaining
REM "if A if B (...) else (...)" directly is unreliable in batch, since the
REM else only binds to the second if.
set TEST_OK=true
if not "!EXITCODE!"=="0" set TEST_OK=false
if "!FOUND_ERROR_MARKER!"=="true" set TEST_OK=false

if "!TEST_OK!"=="true" (
    echo   PASS
    echo PASS  %TEST_NAME%  ^(exit code 0^) >> "%REPORT%"
    set /a PASS_COUNT+=1
) else (
    echo   FAIL  ^(exit code !EXITCODE!, error marker found: !FOUND_ERROR_MARKER!^)
    echo FAIL  %TEST_NAME%  ^(exit code !EXITCODE!, error marker found: !FOUND_ERROR_MARKER!^) - see %LOGFILE% >> "%REPORT%"
    set /a FAIL_COUNT+=1
)

exit /b 0