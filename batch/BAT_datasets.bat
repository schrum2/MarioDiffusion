cd ..

:: These commands convert raw level data into JSON format for each Mario game
python create_level_json_data.py --output "datasets\\SMB1_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros\\Processed"
python create_level_json_data.py --output "datasets\\SMB2_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros 2 (Japan)\\Processed"
python create_level_json_data.py --output "datasets\\SML_Levels.json"  --levels "..\\TheVGLC\\Super Mario Land\\Processed"

:: These commands merge multiple JSON datasets into larger ones
python combine_data.py datasets\\Mario_Levels.json    datasets\\SMB1_Levels.json datasets\\SMB2_Levels.json datasets\\SML_Levels.json
python combine_data.py datasets\\SMB1AND2_Levels.json datasets\\SMB1_Levels.json datasets\\SMB2_Levels.json 

:: For each dataset, captions are generated in two modes: regular and with absence descriptions
default_out1=datasets\\SMB1_LevelsAndCaptions
default_out2=datasets\\SMB2_LevelsAndCaptions
default_out3=datasets\\SML_LevelsAndCaptions
default_out4=datasets\\Mario_LevelsAndCaptions
default_out5=datasets\\SMB1AND2_LevelsAndCaptions
python create_ascii_captions.py --dataset datasets\\SMB1_Levels.json --output %default_out1%-regular.json
python create_ascii_captions.py --dataset datasets\\SMB1_Levels.json --output %default_out1%-absence.json --describe_absence
python create_ascii_captions.py --dataset datasets\\SMB2_Levels.json --output %default_out2%-regular.json
python create_ascii_captions.py --dataset datasets\\SMB2_Levels.json --output %default_out2%-absence.json --describe_absence
python create_ascii_captions.py --dataset datasets\\SML_Levels.json  --output %default_out3%-regular.json
python create_ascii_captions.py --dataset datasets\\SML_Levels.json  --output %default_out3%-absence.json --describe_absence
python create_ascii_captions.py --dataset datasets\\Mario_Levels.json --output %default_out4%-regular.json
python create_ascii_captions.py --dataset datasets\\Mario_Levels.json --output %default_out4%-absence.json --describe_absence
python create_ascii_captions.py --dataset datasets\\SMB1AND2_Levels.json --output %default_out5%-regular.json
python create_ascii_captions.py --dataset datasets\\SMB1AND2_Levels.json --output %default_out5%-absence.json --describe_absence

:: These commands save the tokenized data for each dataset into pickle files
python tokenizer.py save --json_file %default_out1%-regular.json --pkl_file datasets\\SMB1_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out1%-absence.json --pkl_file datasets\\SMB1_Tokenizer-absence.pkl
python tokenizer.py save --json_file %default_out2%-regular.json --pkl_file datasets\\SMB2_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out2%-absence.json --pkl_file datasets\\SMB2_Tokenizer-absence.pkl
python tokenizer.py save --json_file %default_out3%-regular.json  --pkl_file datasets\\SML_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out3%-absence.json  --pkl_file datasets\\SML_Tokenizer-absence.pkl
python tokenizer.py save --json_file %default_out4%-regular.json    --pkl_file datasets\\Mario_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out4%-absence.json    --pkl_file datasets\\Mario_Tokenizer-absence.pkl
python tokenizer.py save --json_file %default_out5%-regular.json --pkl_file datasets\\SMB1AND2_Tokenizer-regular.pkl
python tokenizer.py save --json_file %default_out5%-absence.json --pkl_file datasets\\SMB1AND2_Tokenizer-absence.pkl

:: These commands create validation captions for each dataset, using the previously generated JSON files
:: Added the --no_upside_down_pipes flag to SMB1 validation captions
python create_validation_captions.py --save_file "datasets\\SMB1_ValidationCaptions-regular.json" --json %default_out1%-regular.json --seed 0 --no_upside_down_pipes
python create_validation_captions.py --save_file "datasets\\SMB1_ValidationCaptions-absence.json" --json %default_out1%-absence.json --seed 0 --describe_absence --no_upside_down_pipes
python create_validation_captions.py --save_file "datasets\\SMB2_ValidationCaptions-regular.json" --json %default_out2%-regular.json --seed 0
python create_validation_captions.py --save_file "datasets\\SMB2_ValidationCaptions-absence.json" --json %default_out2%-absence.json --seed 0 --describe_absence
python create_validation_captions.py --save_file "datasets\\SML_ValidationCaptions-regular.json" --json %default_out3%-regular.json --seed 0
python create_validation_captions.py --save_file "datasets\\SML_ValidationCaptions-absence.json" --json %default_out3%-absence.json --seed 0 --describe_absence
python create_validation_captions.py --save_file "datasets\\SMB1AND2_ValidationCaptions-regular.json" --json %default_out5%-regular.json --seed 0 
python create_validation_captions.py --save_file "datasets\\SMB1AND2_ValidationCaptions-absence.json" --json %default_out5%-absence.json --seed 0 --describe_absence
python create_validation_captions.py --save_file "datasets\\Mario_ValidationCaptions-regular.json" --json %default_out4%-regular.json --seed 0
python create_validation_captions.py --save_file "datasets\\Mario_ValidationCaptions-absence.json" --json %default_out4%-absence.json --seed 0 --describe_absence

:: Split output files into train/val/test sets
python split_data.py --json %default_out1%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out1%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out2%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out2%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out3%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out3%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out4%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out4%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out5%-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42
python split_data.py --json %default_out5%-absence.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42