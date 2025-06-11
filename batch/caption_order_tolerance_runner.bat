REM Loop through all directories starting with Mar1and2-conditional in the current directory and run evaluate_caption_order_tolerance
for /D %%D in ("Mar1and2-conditional*0") do (
    if "absence" in %%D(
        set TYPE="absence"
    ) else (
        set TYPE="regular"
    )

    python evaluate_caption_order_tolerance.py --json datasets\Mar1and2_LevelsAndCaptions-%TYPE%-test.json --game Mario --save_as_json --model_path %%~nxD
)
pause