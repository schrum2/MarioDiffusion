cd ..

REM AMED REAL
python evaluate_models.py --modes real random short real_full --full_metrics --metric average_min_edit_distance_from_real --plot_label "Edit Distance" --save --output_name "AMED-REAL_real(full)_real(100)_random_unconditional"

python evaluate_models.py --modes real random short --metric average_min_edit_distance_from_real --plot_label "Edit Distance" --save --output_name "AMED-REAL_real(100)_random_unconditional"


REM AMED SELF
python evaluate_models.py --modes real random short real_full --full_metrics --metric average_min_edit_distance --plot_label "Edit Distance" --save --output_name "AMED-SELF_real(full)_real(100)_random_unconditional"

python evaluate_models.py --modes real random short --metric average_min_edit_distance --plot_label "Edit Distance" --save --output_name "AMED-SELF_real(100)_random_unconditional"


REM Pipe Metrics
python evaluate_models.py --modes real random short real_full --full_metrics --metric broken_pipes_percentage_in_dataset --plot_label "Percent Broken Pipes" --save --output_name "BPPDataset_real(full)_real(100)_random_unconditional"

python evaluate_models.py --modes real random short --metric broken_pipes_percentage_in_dataset --plot_label "Percent Broken Pipes" --save --output_name "BPPDataset_real(100)_random_unconditional"

python evaluate_models.py --modes real random short real_full --full_metrics --metric broken_pipes_percentage_of_pipes --plot_label "Percent Broken Pipes" --save --output_name "BPPPipes_real(full)_real(100)_random_unconditional"

python evaluate_models.py --modes real random short --metric broken_pipes_percentage_of_pipes --plot_label "Percent Broken Pipes" --save --output_name "BPPPipes_real(100)_random_unconditional"


REM Cannon Metrics
python evaluate_models.py --modes real random short real_full --full_metrics --metric broken_cannons_percentage_in_dataset --plot_label "Percent Broken Cannons" --save --output_name "BCPDataset_real(full)_real(100)_random_unconditional"

python evaluate_models.py --modes real random short --metric broken_cannons_percentage_in_dataset --plot_label "Percent Broken Cannons" --save --output_name "BCPDataset_real(100)_random_unconditional"

python evaluate_models.py --modes real random short real_full --full_metrics --metric broken_cannons_percentage_of_cannons --plot_label "Percent Broken Cannons" --save --output_name "BCPCannons_real(full)_real(100)_random_unconditional"

python evaluate_models.py --modes real random short --metric broken_cannons_percentage_of_cannons --plot_label "Percent Broken Cannons" --save --output_name "BCPCannons_real(100)_random_unconditional"
