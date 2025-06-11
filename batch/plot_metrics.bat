cd ..

python evaluate_models.py --modes real random short --metric average_min_edit_distance
python evaluate_models.py --modes real random short --metric broken_pipes_percentage_in_dataset
python evaluate_models.py --modes real random short --metric broken_pipes_percentage_of_pipes
python evaluate_models.py --modes real random short --metric broken_cannons_percentage_in_dataset
python evaluate_models.py --modes real random short --metric broken_cannons_percentage_of_cannons
python evaluate_models.py --modes real random short --metric average_min_edit_distance_from_real
python evaluate_models.py --modes real random short --metric generated_vs_real_perfect_matches
python evaluate_models.py --modes real random short --metric percent_perfect_matches
python evaluate_models.py --modes real random short --metric broken_cannons_percentage_of_cannons
python evaluate_models.py --modes real random short --metric broken_cannons_percentage_of_cannons

python evaluate_models.py --modes real random --metric perfect_match_percentage
python evaluate_models.py --modes real random --metric perfect_match_count
python evaluate_models.py --modes real random --metric partial_match_percentage
python evaluate_models.py --modes real random --metric partial_match_count
python evaluate_models.py --modes real random --metric no_match_percentage
python evaluate_models.py --modes real random --metric no_match_count

python evaluate_models.py --modes long --metric average_min_edit_distance
python evaluate_models.py --modes long --metric broken_pipes_percentage_in_dataset
python evaluate_models.py --modes long --metric broken_pipes_percentage_of_pipes
python evaluate_models.py --modes long --metric broken_cannons_percentage_in_dataset
python evaluate_models.py --modes long --metric broken_cannons_percentage_of_cannons