REM Training time evaluation and visualization
:: Calculate total training time and best epoch training time for each model
call calculate_times.bat
:: Aggregate and format the output for visualization
python evaluate_execution_time.py
:: BAR plot for mean grouped runtimes for each model with standard error for individual times
python visualize_best_model_stats.py --input training_runtimes\\mean_grouped_runtimes_plus_best.json --output total_time_with_std_err.pdf --plot_type bar --y_axis "group" --x_axis "mean" --x_axis_label "Hours to Train" --convert_time_to_hours --stacked_bar_for_mlm 
:: BOX plot for individual times by model
python visualize_best_model_stats.py --input mean_grouped_runtimes.json --output mean_grouped_runtime_box_plot.pdf --plot_type box --x_axis "group" --y_axis "individual_times" --x_axis_label "Models" --y_axis_label "Hours to Train" --convert_time_to_hours --x_tick_rotation 45

REM Best Model Statistics
:: Aggregate and format best_model_info.json info from each model (created while training) for visualizationGather
python best_model_statistics.py
:: This will return: skipped_model_dirs_{date}.json (this should be an empty list if everything was processed), 
:: best_mlm_model_info_{date}.json, and best_model_info_{date}.json 
:: BOX PLOT for best epoch
python visualize_best_model_stats.py --input best_model_info_20250618_162147.json --output best_epoch_box_plot.pdf --plot_type box --x_axis "group" --y_axis "best_epoch" --x_axis_label "Models" --y_axis_label "Best Epoch" --x_tick_rotation 45 
: VIOLIN PLOT for best epoch
python visualize_best_model_stats.py --input best_model_info_20250618_162147.json --output best_epoch_violin_plot.pdf --plot_type violin --x_axis "group" --y_axis "best_epoch" --y_axis_label "Best Epoch" --x_tick_rotation 45 
:: BAR PLOT for best epoch
python visualize_best_model_stats.py --input best_model_info_20250618_162147.json --output best_epoch_bar_plot.pdf --plot_type bar --y_axis "group" --x_axis "best_epoch" --x_axis_label "Best Epoch" --x_markers_on_bar_plot 
:: SCATTER PLOT for best caption x best epoch
python visualize_best_model_stats.py --input best_model_info_20250618_162147.json --output best_epoch_v_best_caption_score_scatter_plot.pdf --plot_type scatter --y_axis "best_caption_score" --x_axis "best_epoch" --y_axis_label "Best Caption Score" --x_axis_label "Best Epoch" 