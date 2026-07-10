@echo off
REM Usage: MM2-extract.bat [limit] [likes]
REM Pulls raw MM2 levels from HuggingFace, decodes them with toost, and writes ASCII levels.
REM [limit] is optional, defaults to 10000
REM [likes] is optional, defaults to 10
cd ..

set LIMIT=%1
if "%LIMIT%"=="" set LIMIT=10000

set LIKES=%2
if "%LIKES%"=="" set LIKES=10

REM Download raw .bcd levels from HuggingFace
python -m mm2pipeline_data extract --output_folder MM2_Data\bcd --limit %LIMIT% --skip_3dworld --skip_items --skip_subworld_items --likes %LIKES%

REM Decode .bcd into JSON and preview images with toost
python -m mm2pipeline_data toost --input MM2_Data\bcd -o MM2_Data\json --images-output MM2_Data\images

REM Convert decoded JSON levels into ASCII tile grids
python -m mm2pipeline_data json-to-ascii --input MM2_Data\json --output_folder MM2_Data\ascii
