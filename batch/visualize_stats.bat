cd ..

:: Calculate and visualize mean and individual runtimes from mean_grouped_runtimes.json
cd batch
call calculate_times.bat
cd ..
python evaluate_execution_time.py

:: BAR plot for mean grouped runtimes with individual times overlayed as a scatter plot
python visualize_best_model_stats.py --input mean_grouped_runtimes.json --output mean_grouped_runtime_bar_plot.pdf --plot_type bar --y_axis "group" --x_axis "mean" --x_axis_label "Hours to Train" --convert_time_to_hours --stacked_bar_for_mlm --x_markers_on_bar_plot --x_marker_data_on_bar_plot "individual_times"
:: BOX plot for individual times by model
python visualize_best_model_stats.py --input mean_grouped_runtimes.json --output mean_grouped_runtime_box_plot.pdf --plot_type box --x_axis "group" --y_axis "individual_times" --x_axis_label "Models" --y_axis_label "Hours to Train" --convert_time_to_hours --x_tick_rotation 45


:: Calculate and visualize best epoch, caption score, and loss
python best_model_statistics.py
:: This will return: skipped_model_dirs_{date}.json (this should be an empty list if everything was processed), 
:: best_mlm_model_info_{date}.json (which we did not plot), 
:: and best_model_info_{date}.json which contains the best epoch, caption score, and loss for each model plotted below.
:: NOTE: Use the best_model_info_{date}.json for the following visualizations

:: BOX PLOT for best epoch
python visualize_best_model_stats.py --input best_model_info_20250618_162147.json --output best_epoch_box_plot.pdf --plot_type box --x_axis "group" --y_axis "best_epoch" --x_axis_label "Models" --y_axis_label "Best Epoch" --x_tick_rotation 45 

: VIOLIN PLOT for best epoch
python visualize_best_model_stats.py --input best_model_info_20250618_162147.json --output best_epoch_violin_plot.pdf --plot_type violin --x_axis "group" --y_axis "best_epoch" --y_axis_label "Best Epoch" --x_tick_rotation 45 

:: BAR PLOT for best epoch
python visualize_best_model_stats.py --input best_model_info_20250618_162147.json --output best_epoch_bar_plot.pdf --plot_type bar --y_axis "group" --x_axis "best_epoch" --x_axis_label "Best Epoch" --x_markers_on_bar_plot 

:: SCATTER PLOT for best caption x best epoch
python visualize_best_model_stats.py --input best_model_info_20250618_162147.json --output best_epoch_v_best_caption_score_scatter_plot.pdf --plot_type scatter --y_axis "best_caption_score" --x_axis "best_epoch" --y_axis_label "Best Caption Score" --x_axis_label "Best Epoch" 




:: Run A* for metric results on conditional models ending in 0
call Mar1and2-conditional-evaluate-solvability.bat
:: TO DO: Write a proper batch script to run A* on all models ending in 0 including wgan, fdm, unconditional, mlm, conditional models, and MarioGPT

:: Visualize A* Solvability
python evaluate_models.py --plot_file astar_result_overall_averages.json --modes real random short --metric "beaten" --plot_label "Percent Beatable Levels" --save