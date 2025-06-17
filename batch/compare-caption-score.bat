cd ..

setlocal enabledelayedexpansion

REM Compare all
for %%d in (LevelsAndCaptions RandomTest) do (
  python plot_average_caption_score.py ^
    "Mar1and2-conditional-regular:0-9:Mar1and2_%%d-regular_scores_by_epoch.jsonl:MLM-regular:0" ^
    "Mar1and2-conditional-absence:0-9:Mar1and2_%%d-absence_scores_by_epoch.jsonl:MLM-absence:1" ^
    "Mar1and2-conditional-negative:0-9:Mar1and2_%%d-regular_scores_by_epoch.jsonl:MLM-negative:2" ^
    "Mar1and2-conditional-MiniLM-regular:0-9:Mar1and2_%%d-regular_scores_by_epoch.jsonl:MiniLM-single-regular:3" ^
    "Mar1and2-conditional-MiniLM-absence:0-9:Mar1and2_%%d-absence_scores_by_epoch.jsonl:MiniLM-single-absence:4" ^
    "Mar1and2-conditional-MiniLM-negative:0-9:Mar1and2_%%d-regular_scores_by_epoch.jsonl:MiniLM-single-negative:5" ^
    "Mar1and2-conditional-MiniLMsplit-regular:0-4:Mar1and2_%%d-regular_scores_by_epoch.jsonl:MiniLM-multiple-regular:6" ^
    "Mar1and2-conditional-MiniLMsplit-absence:0-4:Mar1and2_%%d-absence_scores_by_epoch.jsonl:MiniLM-multiple-absence:7" ^
    "Mar1and2-conditional-MiniLMsplit-negative:0-4:Mar1and2_%%d-regular_scores_by_epoch.jsonl:MiniLM-multiple-negative:8" ^
    "Mar1and2-conditional-GTE-regular:0-4:Mar1and2_%%d-regular_scores_by_epoch.jsonl:GTE-single-regular:9" ^
    "Mar1and2-conditional-GTE-absence:0-4:Mar1and2_%%d-absence_scores_by_epoch.jsonl:GTE-single-absence:10" ^
    "Mar1and2-conditional-GTE-negative:0-4:Mar1and2_%%d-regular_scores_by_epoch.jsonl:GTE-single-negative:11" ^
    "Mar1and2-conditional-GTEsplit-regular:0:Mar1and2_%%d-regular_scores_by_epoch.jsonl:GTE-multiple-regular:12" ^
    "Mar1and2-conditional-GTEsplit-absence:0:Mar1and2_%%d-absence_scores_by_epoch.jsonl:GTE-multiple-absence:13" ^
    "Mar1and2-conditional-GTEsplit-negative:0:Mar1and2_%%d-regular_scores_by_epoch.jsonl:GTE-multiple-negative:14" ^
    "Mar1and2-fdm-MiniLM-regular:0-29:Mar1and2_%%d-regular_scores_by_epoch.jsonl:FDM-MiniLM-regular:15" ^
    "Mar1and2-fdm-MiniLM-absence:0-29:Mar1and2_%%d-absence_scores_by_epoch.jsonl:FDM-MiniLM-absence:16" ^
    "Mar1and2-fdm-GTE-regular:0-29:Mar1and2_%%d-regular_scores_by_epoch.jsonl:FDM-GTE-regular:17" ^
    "Mar1and2-fdm-GTE-absence:0-29:Mar1and2_%%d-absence_scores_by_epoch.jsonl:FDM-GTE-absence:18" ^
    --ci --pdf "CaptionAdherence-%%d.pdf"
)

REM Compare by caption strategy
for %%d in (LevelsAndCaptions RandomTest) do (  for %%t in (regular absence negative) do (
    REM Set base variables
    set DATA=%%t
    if /I "!DATA!"=="negative" set DATA=regular
    
    REM Calculate style index based on condition type:
    REM regular=+0, absence=+1, negative=+2
    set STYLE_INDEX=0
    if /I "%%t"=="absence" set STYLE_INDEX=1
    if /I "%%t"=="negative" set STYLE_INDEX=2
    
    REM Calculate final indices for each model type
    set /a MLM_STYLE=!STYLE_INDEX!
    set /a MINILN_SINGLE_STYLE=!STYLE_INDEX!+3
    set /a MINILN_MULTI_STYLE=!STYLE_INDEX!+6
    set /a GTE_SINGLE_STYLE=!STYLE_INDEX!+9
    set /a GTE_MULTI_STYLE=!STYLE_INDEX!+12
    set /a FDM_MINILN_STYLE=!STYLE_INDEX!+15
    set /a FDM_GTE_STYLE=!STYLE_INDEX!+17
    
    python plot_average_caption_score.py ^
        "Mar1and2-conditional-%%t:0-9:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:MLM-%%t:!MLM_STYLE!" ^
        "Mar1and2-conditional-MiniLM-%%t:0-9:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:MiniLM-single-%%t:!MINILN_SINGLE_STYLE!" ^
        "Mar1and2-conditional-MiniLMsplit-%%t:0-4:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:MiniLM-multiple-%%t:!MINILN_MULTI_STYLE!" ^
        "Mar1and2-conditional-GTE-%%t:0-4:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:GTE-single-%%t:!GTE_SINGLE_STYLE!" ^
        "Mar1and2-conditional-GTEsplit-%%t:0:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:GTE-multiple-%%t:!GTE_MULTI_STYLE!" ^
        "Mar1and2-fdm-MiniLM-!DATA!:0-29:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:FDM-MiniLM-!DATA!:!FDM_MINILN_STYLE!" ^
        "Mar1and2-fdm-GTE-!DATA!:0-29:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:FDM-GTE-!DATA!:!FDM_GTE_STYLE!" ^
        --ci --pdf "CaptionAdherence-%%d-%%t.pdf"
  )
)


