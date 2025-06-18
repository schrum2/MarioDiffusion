cd ..

:: Visualize mean and individual runtimes from mean_grouped_runtimes.json
:: BAR plot for mean grouped runtimes with individual times overlayed as a scatter plot
python visualize_best_model_stats.py --input mean_grouped_runtimes.json --output mean_grouped_runtime_bar_plot.pdf --plot_type bar --y_axis "group" --x_axis "mean" --x_axis_label "Hours to Train" --convert_time_to_hours --stacked_bar_for_mlm --x_markers_on_bar_plot --x_marker_data_on_bar_plot "individual_times"
:: BOX plot for individual times by model
python visualize_best_model_stats.py --input mean_grouped_runtimes.json --output mean_grouped_runtime_box_plot.pdf --plot_type box --x_axis "group" --y_axis "individual_times" --x_axis_label "MODELS" --y_axis_label "Hours to Train" --convert_time_to_hours --x_tick_rotation 45



:: Visualize best epoch, caption score, and loss from best_model_statistics.jsonl
:: BOX best epoch
python visualize_best_model_stats.py --input best_model_statistics.jsonl --output test_box_plot.pdf --plot_type box --x_axis "group" --y_axis "best_epoch" --x_axis_label "MODELS" --y_axis_label "BEST EPOCH" --x_tick_rotation 45 

: VIOLIN best epoch
python visualize_best_model_stats.py --input best_model_statistics.jsonl --output test_violin_plot.pdf --plot_type violin --x_axis "group" --y_axis "best_epoch" --y_axis_label "BEST EPOCH" --x_tick_rotation 45 

:: BAR best epoch
python visualize_best_model_stats.py --input best_model_statistics.jsonl --output test_bar_plot.pdf --plot_type bar --y_axis "group" --x_axis "best_epoch" --x_axis_label "BEST EPOCH" --x_markers_on_bar_plot 

:: SCATTER caption x epoch
python visualize_best_model_stats.py --input best_model_statistics.jsonl --output test_scatter_plot.pdf --plot_type scatter --y_axis "best_caption_score" --x_axis "best_epoch" --y_axis_label "BEST CAPTION SCORE" --x_axis_label "BEST EPOCH" 

:: Visualize A* Solvability
python evaluate_models.py --plot_file astar_result_overall_averages.json --modes real random short --metric "beaten" --plot_label "Percent Beatable Levels" --save