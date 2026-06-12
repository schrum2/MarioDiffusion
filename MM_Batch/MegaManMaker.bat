@echo off

setlocal enabledelayedexpansion

set MMM=%USERPROFILE%\AppData\Local\MegaMaker\Levels

cd /d "%~dp0.."

set INTERACTIVE=1
if not "%~1"=="" (
    set INTERACTIVE=0
    set FILE=%~1
    call :GETINFO "%~1"
    if "!EXT!"==".mmlv" goto DOMMLV
    if "!EXT!"==".txt" goto DOTXT
    echo Unsupported file type.
    goto END
)

:LOOP
set FILE=
set /p FILE=Upload a file (or type Q to quit): 

if /i "!FILE!"=="q" goto END

call :GETINFO "!FILE!"

if "!EXT!"==".mmlv" goto DOMMLV
if "!EXT!"==".txt" goto DOTXT
echo Unsupported file type. Please provide a .mmlv or .txt file.
goto LOOP

:DOMMLV
python -m megaman.mmlv_to_vglc "!FILE!"
if "!INTERACTIVE!"=="0" goto END
goto LOOP

:DOTXT
python -m megaman.vglc_to_mmlv "!FILE!" "!DIR!!NAME!.mmlv"
copy "!DIR!!NAME!.mmlv" "%MMM%\"
if "!INTERACTIVE!"=="0" goto END
goto LOOP

:GETINFO
set NAME=%~n1
set DIR=%~dp1
set EXT=%~x1
exit /b

:END