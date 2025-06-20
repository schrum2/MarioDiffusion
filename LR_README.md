## Can I also get Lode Runner Data? (TODO)

This repository is needed to be able to play the Lode Runner levels, but you must fork at the link https://github.com/williamsr03/LodeRunner and then clone repo with your own username. This should create a folder with the MarioDiffusion folder that contains all needed Lode Runner items:
```
git clone --branch Reid --single-branch https://github.com/williamsr03/LodeRunner.git
```
Next, enter the `LodeRunner` respository.
```
cd LodeRunner
```
Then install Lode Runner repository as a library so it can be used with the MarioDiffusion data:
```
pip install -e .
```
Then go back to the MarioDiffusion directory to be able to run the following.

Extract a json data set of 32 by 32 level scenes from the VGLC data for Lode Runner with a command like this (top 10 rows are filled with blank space to make a perfect square):
```
python create_level_json_data.py --output "LR_Levels.json" --levels "..\\TheVGLC\\Lode Runner\\Processed" --tileset "..\\TheVGLC\\Lode Runner\\Loderunner.json" --target_height 32 --target_width 32 --extra_tile .
```

These files only contains the level scenes. Create captions for all level scenes with a command like this:
```
python LR_create_ascii_captions.py --dataset LR_Levels.json --output LR_LevelsAndCaptions-regular.json
```

You can also make the captions explicitly mention things that are absent from each scene with the `--describe_absence` flag:
```
python LR_create_ascii_captions.py --dataset LR_Levels.json --output LR_LevelsAndCaptions-absence.json --describe_absence
```

Browse LR data with ascii browser and be able to play some of the Lode Runner levels:
```
python ascii_data_browser.py LR_LevelsAndCaptions-regular.json "..\\TheVGLC\\Lode Runner\\Loderunner.json"
```

To train an unconditional diffusion model without any text embeddings, run this command:
```
python train_diffusion.py --augment --output_dir "LR-unconditional-regular" --num_epochs 100 --json LR_LevelsAndCaptions-regular.json --split --num_tiles 10 --batch_size 5 --game LR
```

Run trained unconditional diffusion model and save 100 random levels to json:
```
python run_diffusion.py --model_path LR-unconditional-regular --num_samples 100 --save_as_json --output_dir "LR-unconditional-regular-samples" --game LR
```

First create a tokenizer for the caption data you want to train on. Most of these datasets have the same vocabulary, but there is a clear difference between datasets that describe the absence of entities and those that do not.
```
python tokenizer.py save --json_file LR_LevelsAndCaptions-regular.json --pkl_file LR_Tokenizer-regular.pkl
```

If the user wanted to play the levels, use the following command line. The following line allows the user to play the first level. If the user wants to play a different level, change the 1 to the level they wish to play.
Must be in a directory that contains both of the other two directory before using this command line.
```
python LodeRunner\loderunner\main.py MarioDiffusion/LR_LevelsAndCaptions-regular.json 1
```

Batch file that fully runs unconditional diffusion for Lode Runner (as long as the file do not exist):
```
BAT_LR-unconditional.bat
```

Batch file that fully runs conditional diffusion for Lode Runner (as long as the file do not exist):
```
BAT_LR-conditional.bat
```
