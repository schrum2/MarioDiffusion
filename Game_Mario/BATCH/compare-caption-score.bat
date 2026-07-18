@echo off
setlocal enabledelayedexpansion
pushd "%~dp0\..\.."

set MODEL_PREFIX=Mario-Mar1and2

REM Compare all
for %%d in (LevelsAndCaptions RandomTest test) do (

  set REGULAR_FILE=Mar1and2_%%d-regular_scores_by_epoch.jsonl
  set ABSENCE_FILE=Mar1and2_%%d-absence_scores_by_epoch.jsonl

  REM Set y-axis limits based on dataset
  set YMIN=0.0
  set YMAX=1.0
  if /I "%%d"=="RandomTest" (
    set YMIN=-0.2
    set YMAX=0.47
  )

  if /I "%%d"=="test" (
    set REGULAR_FILE=Mar1and2_LevelsAndCaptions-regular-test_scores_by_epoch.jsonl
    set ABSENCE_FILE=Mar1and2_LevelsAndCaptions-absence-test_scores_by_epoch.jsonl
  )

  python plot_average_caption_score.py ^
    "!MODEL_PREFIX!-conditional-regular:0-9:!REGULAR_FILE!:MLM-regular:0" ^
    "!MODEL_PREFIX!-conditional-absence:0-9:!ABSENCE_FILE!:MLM-absence:1" ^
    "!MODEL_PREFIX!-conditional-negative:0-9:!REGULAR_FILE!:MLM-negative:2" ^
    "!MODEL_PREFIX!-conditional-MiniLM-regular:0-9:!REGULAR_FILE!:MiniLM-single-regular:3" ^
    "!MODEL_PREFIX!-conditional-MiniLM-absence:0-9:!ABSENCE_FILE!:MiniLM-single-absence:4" ^
    "!MODEL_PREFIX!-conditional-MiniLM-negative:0-9:!REGULAR_FILE!:MiniLM-single-negative:5" ^
    "!MODEL_PREFIX!-conditional-MiniLMsplit-regular:0-4:!REGULAR_FILE!:MiniLM-multiple-regular:6" ^
    "!MODEL_PREFIX!-conditional-MiniLMsplit-absence:0-4:!ABSENCE_FILE!:MiniLM-multiple-absence:7" ^
    "!MODEL_PREFIX!-conditional-MiniLMsplit-negative:0-4:!REGULAR_FILE!:MiniLM-multiple-negative:8" ^
    "!MODEL_PREFIX!-conditional-GTE-regular:0-4:!REGULAR_FILE!:GTE-single-regular:9" ^
    "!MODEL_PREFIX!-conditional-GTE-absence:0-4:!ABSENCE_FILE!:GTE-single-absence:10" ^
    "!MODEL_PREFIX!-conditional-GTE-negative:0-4:!REGULAR_FILE!:GTE-single-negative:11" ^
    "!MODEL_PREFIX!-conditional-GTEsplit-regular:0:!REGULAR_FILE!:GTE-multiple-regular:12" ^
    "!MODEL_PREFIX!-conditional-GTEsplit-absence:0:!ABSENCE_FILE!:GTE-multiple-absence:13" ^
    "!MODEL_PREFIX!-conditional-GTEsplit-negative:0:!REGULAR_FILE!:GTE-multiple-negative:14" ^
    "!MODEL_PREFIX!-fdm-MiniLM-regular:0-29:!REGULAR_FILE!:FDM-MiniLM-regular:15" ^
    "!MODEL_PREFIX!-fdm-MiniLM-absence:0-29:!ABSENCE_FILE!:FDM-MiniLM-absence:16" ^
    "!MODEL_PREFIX!-fdm-GTE-regular:0-29:!REGULAR_FILE!:FDM-GTE-regular:17" ^
    "!MODEL_PREFIX!-fdm-GTE-absence:0-29:!ABSENCE_FILE!:FDM-GTE-absence:18" ^
    --ci --pdf "CaptionAdherence-%%d.pdf" --ymin !YMIN! --ymax !YMAX!
)

REM Compare by caption strategy
for %%d in (LevelsAndCaptions RandomTest test) do (
  REM Set y-axis limits based on dataset
  set YMIN=0.0
  set YMAX=1.0
  if /I "%%d"=="RandomTest" (
      set YMIN=-0.2
      set YMAX=0.47
  )

  for %%t in (regular absence negative) do (
    REM Set base variables
    set DATA=%%t
    if /I "!DATA!"=="negative" set DATA=regular
    
    REM Calculate style index based on condition type:
    REM regular=+0, absence=+1, negative=+2
    set STYLE_INDEX=0
    if /I "%%t"=="absence" set STYLE_INDEX=1
    if /I "%%t"=="negative" set STYLE_INDEX=2
    
    set DATA_FILE=Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl
    if /I "%%d"=="test" (
      set DATA_FILE=Mar1and2_LevelsAndCaptions-!DATA!-test_scores_by_epoch.jsonl
    )

    REM Calculate final indices for each model type
    set /a MLM_STYLE=!STYLE_INDEX!
    set /a MINILN_SINGLE_STYLE=!STYLE_INDEX!+3
    set /a MINILN_MULTI_STYLE=!STYLE_INDEX!+6
    set /a GTE_SINGLE_STYLE=!STYLE_INDEX!+9
    set /a GTE_MULTI_STYLE=!STYLE_INDEX!+12
    set /a FDM_MINILN_STYLE=!STYLE_INDEX!+15
    set /a FDM_GTE_STYLE=!STYLE_INDEX!+17

    REM Conditionally set FDM_ARGS only if %%t is not negative
    set FDM_ARGS=
    if /I "%%t" NEQ "negative" set FDM_ARGS="!MODEL_PREFIX!-fdm-MiniLM-!DATA!:0-29:!DATA_FILE!:FDM-MiniLM-!DATA!:!FDM_MINILN_STYLE!"  "!MODEL_PREFIX!-fdm-GTE-!DATA!:0-29:!DATA_FILE!:FDM-GTE-!DATA!:!FDM_GTE_STYLE!"

    python plot_average_caption_score.py ^
        "!MODEL_PREFIX!-conditional-%%t:0-9:!DATA_FILE!:MLM-%%t:!MLM_STYLE!" ^
        "!MODEL_PREFIX!-conditional-MiniLM-%%t:0-9:!DATA_FILE!:MiniLM-single-%%t:!MINILN_SINGLE_STYLE!" ^
        "!MODEL_PREFIX!-conditional-MiniLMsplit-%%t:0-4:!DATA_FILE!:MiniLM-multiple-%%t:!MINILN_MULTI_STYLE!" ^
        "!MODEL_PREFIX!-conditional-GTE-%%t:0-4:!DATA_FILE!:GTE-single-%%t:!GTE_SINGLE_STYLE!" ^
        "!MODEL_PREFIX!-conditional-GTEsplit-%%t:0:!DATA_FILE!:GTE-multiple-%%t:!GTE_MULTI_STYLE!" ^
        !FDM_ARGS! --ci --pdf "CaptionAdherence-%%d-%%t.pdf" --ymin !YMIN! --ymax !YMAX!
  )
)

popd
endlocal

