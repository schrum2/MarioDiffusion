## Can I also get Lode Runner Data? (TODO)
This Lode Runner data is still experimental and on-going and the current results are not as good as the Mario
levels and outputs. The main therory as to why is a small dataset with only 150 samples.

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
python LodeRunner\loderunner\main.py LR_LevelsAndCaptions-regular.json 1
```

But to actually provide captions to guide the level generation, use this command
```
python text_to_level_diffusion.py --model_path LR-conditional-regular0 --game LR
```

An easier-to-use GUI interface will let you select and combine known caption phrases to send to the model. Note that the selection of known phrases needs to come from the dataset you trained on.
```
python interactive_tile_level_generator.py LR_LevelsAndCaptions-regular.json LR-conditional-regular0
```

Batch folder that contains all batch files associated with Lode Runner:
```
cd LR_batch
```

Batch file that created regular and absence data associated with Lode Runner:
```
LR-data.bat
```

Batch file that fully runs unconditional diffusion for Lode Runner (as long as the file do not exist):
```
LR-unconditional.bat
```

Batch file that fully runs conditional diffusion for Lode Runner (as long as the file do not exist):
```
LR-conditional.bat
```
