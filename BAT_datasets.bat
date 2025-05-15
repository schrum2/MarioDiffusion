python create_level_json_data.py --output "SMB1_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros\\Processed"
python create_level_json_data.py --output "SMB2_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros 2 (Japan)\\Processed"
python create_level_json_data.py --output "SML_Levels.json"  --levels "..\\TheVGLC\\Super Mario Land\\Processed"

python combine_data.py Mario_Levels.json    SMB1_Levels.json SMB2_Levels.json SML_Levels.json
python combine_data.py SMB1AND2_Levels.json SMB1_Levels.json SMB2_Levels.json 

python create_ascii_captions.py --dataset SMB1_Levels.json --output SMB1_LevelsAndCaptions-regular.json
python create_ascii_captions.py --dataset SMB1_Levels.json --output SMB1_LevelsAndCaptions-absence.json --describe_absence
python create_ascii_captions.py --dataset SMB2_Levels.json --output SMB2_LevelsAndCaptions-regular.json
python create_ascii_captions.py --dataset SMB2_Levels.json --output SMB2_LevelsAndCaptions-absence.json --describe_absence
python create_ascii_captions.py --dataset SML_Levels.json  --output SML_LevelsAndCaptions-regular.json
python create_ascii_captions.py --dataset SML_Levels.json  --output SML_LevelsAndCaptions-absence.json --describe_absence

python create_ascii_captions.py --dataset Mario_Levels.json --output Mario_LevelsAndCaptions-regular.json
python create_ascii_captions.py --dataset Mario_Levels.json --output Mario_LevelsAndCaptions-absence.json --describe_absence
python create_ascii_captions.py --dataset SMB1AND2_Levels.json --output SMB1AND2_LevelsAndCaptions-regular.json
python create_ascii_captions.py --dataset SMB1AND2_Levels.json --output SMB1AND2_LevelsAndCaptions-absence.json --describe_absence

python tokenizer.py save --json_file SMB1_LevelsAndCaptions-regular.json --pkl_file SMB1_Tokenizer-regular.pkl
python tokenizer.py save --json_file SMB1_LevelsAndCaptions-absence.json --pkl_file SMB1_Tokenizer-absence.pkl
python tokenizer.py save --json_file SMB2_LevelsAndCaptions-regular.json --pkl_file SMB2_Tokenizer-regular.pkl
python tokenizer.py save --json_file SMB2_LevelsAndCaptions-absence.json --pkl_file SMB2_Tokenizer-absence.pkl
python tokenizer.py save --json_file SML_LevelsAndCaptions-regular.json  --pkl_file SML_Tokenizer-regular.pkl
python tokenizer.py save --json_file SML_LevelsAndCaptions-absence.json  --pkl_file SML_Tokenizer-absence.pkl

python tokenizer.py save --json_file Mario_LevelsAndCaptions-regular.json    --pkl_file Mario_Tokenizer-regular.pkl
python tokenizer.py save --json_file Mario_LevelsAndCaptions-absence.json    --pkl_file Mario_Tokenizer-absence.pkl
python tokenizer.py save --json_file SMB1AND2_LevelsAndCaptions-regular.json --pkl_file SMB1AND2_Tokenizer-regular.pkl
python tokenizer.py save --json_file SMB1AND2_LevelsAndCaptions-absence.json --pkl_file SMB1AND2_Tokenizer-absence.pkl

python create_validation_captions.py --save_file "Mario_ValidationCaptions-regular.json" --json Mario_LevelsAndCaptions-regular.json --seed 0
python create_validation_captions.py --save_file "Mario_ValidationCaptions-absence.json" --json Mario_LevelsAndCaptions-absence.json --seed 0 --describe_absence