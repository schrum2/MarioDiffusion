cd ..

python evaluate_models.py --modes real random short --metric average_min_edit_distance --save
python evaluate_models.py --modes real random short --metric broken_pipes_percentage_in_dataset --save
python evaluate_models.py --modes real random short --metric broken_pipes_percentage_of_pipes --save
python evaluate_models.py --modes real random short --metric broken_cannons_percentage_in_dataset --save
python evaluate_models.py --modes real random short --metric broken_cannons_percentage_of_cannons --save
python evaluate_models.py --modes real random short --metric average_min_edit_distance_from_real --save
python evaluate_models.py --modes real random short --metric generated_vs_real_perfect_matches --save
python evaluate_models.py --modes real random short --metric percent_perfect_matches --save
python evaluate_models.py --modes real random short --metric broken_cannons_percentage_of_cannons --save
python evaluate_models.py --modes real random short --metric broken_cannons_percentage_of_cannons --save

python evaluate_models.py --modes real random --metric perfect_match_percentage --save
python evaluate_models.py --modes real random --metric perfect_match_count --save
python evaluate_models.py --modes real random --metric partial_match_percentage --save
python evaluate_models.py --modes real random --metric partial_match_count --save
python evaluate_models.py --modes real random --metric no_match_percentage --save
python evaluate_models.py --modes real random --metric no_match_count --save

python evaluate_models.py --modes long --metric average_min_edit_distance --save
python evaluate_models.py --modes long --metric broken_pipes_percentage_in_dataset --save
python evaluate_models.py --modes long --metric broken_pipes_percentage_of_pipes --save
python evaluate_models.py --modes long --metric broken_cannons_percentage_in_dataset --save
python evaluate_models.py --modes long --metric broken_cannons_percentage_of_cannons --save