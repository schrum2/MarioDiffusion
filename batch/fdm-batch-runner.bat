cd ..
set run = 0
set max_run = 5
:loop_start
if !run! GEQ %max_run% goto end

REM Run the Python script with the correct arguments
python evaluate_caption_order_tolerance.py --json datasets\Mar1and2_LevelsAndCaptions-regular-test.json --game Mario --save_as_json --model_path Mar1and2-fdm-GTE-regular0
python evaluate_caption_order_tolerance.py --json datasets\Mar1and2_LevelsAndCaptions-absence-test.json --game Mario --save_as_json --model_path Mar1and2-fdm-GTE-absence0
python evaluate_caption_order_tolerance.py --json datasets\Mar1and2_LevelsAndCaptions-regular-test.json --game Mario --save_as_json --model_path Mar1and2-fdm-MiniLM-regular0
python evaluate_caption_order_tolerance.py --json datasets\Mar1and2_LevelsAndCaptions-absence-test.json --game Mario --save_as_json --model_path Mar1and2-fdm-MiniLM-absence0

set /a run+=1
goto loop_start
:end
python caption_order_stats_creator.py --dir Mar1and2-fdm-GTE-regular0
python caption_order_stats_creator.py --dir Mar1and2-fdm-GTE-absence0
python caption_order_stats_creator.py --dir Mar1and2-fdm-MiniLM-regular0
python caption_order_stats_creator.py --dir Mar1and2-fdm-MiniLM-absence0

python visualize_best_model_stats.py --input caption_order_stats0.jsonl --plot_type horizontal_box --y_axis "group" --x_axis "Lists of average scores" --output "caption_order_tolerance_all_average_boxplot.pdf" --x_axis_label "Caption Order Tolerance Score"
exit /b
