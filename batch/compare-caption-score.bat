cd ..

setlocal enabledelayedexpansion

REM Compare all
for %%d in (LevelsAndCaptions RandomTest) do (
  python plot_average_caption_score.py ^
    "Mar1and2-conditional-regular:0-9:Mar1and2_%%d-regular_scores_by_epoch.jsonl:MLM-regular" ^
    "Mar1and2-conditional-absence:0-9:Mar1and2_%%d-absence_scores_by_epoch.jsonl:MLM-absence" ^
    "Mar1and2-conditional-negative:0-9:Mar1and2_%%d-regular_scores_by_epoch.jsonl:MLM-negative" ^
    "Mar1and2-conditional-MiniLM-regular:0-9:Mar1and2_%%d-regular_scores_by_epoch.jsonl:MiniLM-single-regular" ^
    "Mar1and2-conditional-MiniLM-absence:0-9:Mar1and2_%%d-absence_scores_by_epoch.jsonl:MiniLM-single-absence" ^
    "Mar1and2-conditional-MiniLM-negative:0-9:Mar1and2_%%d-regular_scores_by_epoch.jsonl:MiniLM-single-negative" ^
    "Mar1and2-conditional-MiniLMsplit-regular:0-4:Mar1and2_%%d-regular_scores_by_epoch.jsonl:MiniLM-multiple-regular" ^
    "Mar1and2-conditional-MiniLMsplit-absence:0-4:Mar1and2_%%d-absence_scores_by_epoch.jsonl:MiniLM-multiple-absence" ^
    "Mar1and2-conditional-MiniLMsplit-negative:0-4:Mar1and2_%%d-regular_scores_by_epoch.jsonl:MiniLM-multiple-negative" ^
    "Mar1and2-conditional-GTE-regular:0-4:Mar1and2_%%d-regular_scores_by_epoch.jsonl:GTE-single-regular" ^
    "Mar1and2-conditional-GTE-absence:0-4:Mar1and2_%%d-absence_scores_by_epoch.jsonl:GTE-single-absence" ^
    "Mar1and2-conditional-GTE-negative:0-4:Mar1and2_%%d-regular_scores_by_epoch.jsonl:GTE-single-negative" ^
    "Mar1and2-conditional-GTEsplit-regular:0:Mar1and2_%%d-regular_scores_by_epoch.jsonl:GTE-multiple-regular" ^
    "Mar1and2-conditional-GTEsplit-absence:0:Mar1and2_%%d-absence_scores_by_epoch.jsonl:GTE-multiple-absence" ^
    "Mar1and2-conditional-GTEsplit-negative:0:Mar1and2_%%d-regular_scores_by_epoch.jsonl:GTE-multiple-negative" ^
    "Mar1and2-fdm-MiniLM-regular:0-29:Mar1and2_%%d-regular_scores_by_epoch.jsonl:FDM-MiniLM-regular" ^
    "Mar1and2-fdm-MiniLM-absence:0-29:Mar1and2_%%d-absence_scores_by_epoch.jsonl:FDM-MiniLM-absence" ^
    "Mar1and2-fdm-GTE-regular:0-29:Mar1and2_%%d-regular_scores_by_epoch.jsonl:FDM-GTE-regular" ^
    "Mar1and2-fdm-GTE-absence:0-29:Mar1and2_%%d-absence_scores_by_epoch.jsonl:FDM-GTE-absence" ^
    --ci
)

REM Compare by caption strategy
for %%d in (LevelsAndCaptions RandomTest) do (  for %%t in (regular absence negative) do (
    set DATA=%%t
    if /I "!DATA!"=="negative" set DATA=regular
    python plot_average_caption_score.py ^
      "Mar1and2-conditional-%%t:0-9:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:MLM-%%t" ^      "Mar1and2-conditional-MiniLM-%%t:0-9:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:MiniLM-single-%%t" ^
      "Mar1and2-conditional-MiniLMsplit-%%t:0-4:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:MiniLM-multiple-%%t" ^
      "Mar1and2-conditional-GTE-%%t:0-4:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:GTE-single-%%t" ^
      "Mar1and2-conditional-GTEsplit-%%t:0:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:GTE-multiple-%%t" ^
      "Mar1and2-fdm-MiniLM-!DATA!:0-29:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:FDM-MiniLM-!DATA!" ^
      "Mar1and2-fdm-GTE-!DATA!:0-29:Mar1and2_%%d-!DATA!_scores_by_epoch.jsonl:FDM-GTE-!DATA!" ^
      --ci
  )
)


