cd ..

REM AMED REAL
python evaluate_models.py --modes real random short real_full --full_metrics --metric average_min_edit_distance_from_real --plot_label "Edit Distance" --save --output_name "AMED-REAL_real(full)_real(100)_random_unconditional" --loc "best" --legend_cols 1 --errorbar

REM AMED SELF
REM Create the dataset needed to do this. Call evaluate_metrics.py with the --real_data flag set
python evaluate_metrics.py --real_data --model_path None

python evaluate_models.py --modes real random short real_full --full_metrics --metric average_min_edit_distance --plot_label "Edit Distance" --save --output_name "AMED-SELF_real(full)_real(100)_random_unconditional" --loc "lower right" --bbox 1.0 0.1 --errorbar

REM Pipe Metrics
python evaluate_models.py --modes real random short real_full --full_metrics --metric broken_pipes_percentage_in_dataset --plot_label "Percent Broken/Total Pipe Scenes" --save --output_name "BPPDataset_real(full)_real(100)_random_unconditional" --loc "lower right" --legend_cols 1 --bbox 1.0 0.15 --errorbar

python evaluate_models.py --modes real random short real_full --full_metrics --metric broken_pipes_percentage_of_pipes --plot_label "Percent Broken Pipes" --save --output_name "BPPPipes_real(full)_real(100)_random_unconditional" --loc "lower right" --legend_cols 2 --errorbar

REM Cannon Metrics
python evaluate_models.py --modes real random short real_full --full_metrics --metric broken_cannons_percentage_in_dataset --plot_label "Percent Broken/Total Cannon Scenes" --save --output_name "BCPDataset_real(full)_real(100)_random_unconditional" --loc "lower right" --errorbar

python evaluate_models.py --modes real random short real_full --full_metrics --metric broken_cannons_percentage_of_cannons --plot_label "Percent Broken Cannons" --save --output_name "BCPCannons_real(full)_real(100)_random_unconditional" --loc "lower right" --errorbar

REM A* Solvability
:: Run A* for metric results to create: astar_result.json and astar_result_overall_averages.json for each model
REM call evaluate-solvability.bat
:: Plot A* Solvability
python evaluate_models.py --plot_file astar_result_overall_averages.json --modes real random short --metric "beaten" --plot_label "Percent Beatable Levels" --save