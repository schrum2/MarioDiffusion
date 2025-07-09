# Lode Runner Generation

Generate Lode Runner level scenes with a diffusion model conditioned on text input.
This Lode Runner data is still experimental and on-going and the current results are not as good as the Mario levels and outputs. The main therory as to why is a small dataset with only 150 samples.

## Set up the repository

Everything needed for playing Lode Runner can be accessed be rerunning the `requirements.txt` file:
```
pip uninstall loderunner
```
Before running any code, install all requirements with pip:
```
pip install -r requirements.txt
```
Before being able to play some Lode Runner levels, you must create a dataset which happens below.
## Create datasets

Data used for training our models already exists in the `datasets` directory of this repo,
but you can recreate the data using these commands. First, you will need to check out 
[my forked copy of TheVGLC](https://github.com/schrum2/TheVGLC). Note that the following
command should be executed in the parent directory of the `MarioDiffusion` repository so that
the directories for `MarioDiffusion` and `TheVGLC` are next to each other in the same directory:
```
git clone https://github.com/schrum2/TheVGLC.git
```
Once you have my version of `TheVGLC` and `MarioDiffusion`, go into the `LR_batch` sub-directory in the
`MarioDiffusion` repo for Lode Runner batch files.
```
cd MarioDiffusion
cd LR_batch
```
Next, run a batch file to create datasets from the VGLC data. This batch file call will create
a json data set of 32 by 32 level scenes from the VGLC data for Lode Runner with a command like this 
(top 10 rows are filled with blank space to make a perfect square).
Afterwards, it will create captions for the dataset, tokenizers for the data, random test captions for later evaluation, and finally splits the data into training, validation, and testing json files. 
These files will overwrite the files already in the repo, but they should be identical.
Run this command:

```
LR-data.bat
```

Now you can browse level scenes and their captions with a command like this (the json file can be replaced by any levels and captions json file in datasets):
```
python ascii_data_browser.py datasets\LR_LevelsAndCaptions-regular.json datasets\Loderunner.json
```

## Complete training and evaluation sequence

To train and run an unconditional diffusion model without any text embeddings, go within the 
`LR_batch` sub-directory:
```
cd LR_batch
```
Run this command:
```
LR-unconditional.bat 0 regular
```

To train and run a conditional diffusion model without any text embeddings, go within the 
`LR_batch` sub-directory:
```
cd LR_batch
```
The following command trains an MLM model on the Lode Runner data, trains a conditional diffusion model,
runs the diffusion model, and evaluates the caption adherence based on the generated levels and captions:
```
LR-conditional.bat 0 regular
```
## Generating and playing Lode Runner levels
If the user wants to see the captions and play all of the original levels, use the following command line:
```
python ascii_data_browser.py datasets\LR_LevelsAndCaptions-regular.json datasets\Loderunner.json
```

If the user wanted to play the levels, use the following command line. The following line allows the user to play the first level. If the user wants to play a different level, change the 1 to the level they wish to play.
Must be in a directory that contains both of the other two directory before using this command line.
```
python -m loderunner.main datasets\LR_LevelsAndCaptions-regular.json 1
```

But to actually provide captions to guide the level generation, use this command
```
python text_to_level_diffusion.py --model_path LR-conditional-regular0 --game LR
```

An easier-to-use GUI interface will let you select and combine known caption phrases to send to the model. Note that the selection of known phrases needs to come from the dataset you trained on.
```
python interactive_tile_level_generator.py --load_data datasets\LR_LevelsAndCaptions-regular.json --model_path LR-conditional-regular0 --game LR 
```

## Batch folder and files with Lode Runner
Batch folder that contains all batch files associated with Lode Runner:
```
cd LR_batch
```

Batch file that created regular and absence data associated with Lode Runner:
```
LR-data.bat
```

Batch file that fully trains and runs a unconditional diffusion model for Lode Runner (as long as the file do not exist):
```
LR-unconditional.bat
```

Batch file that fully trains and runs a conditional diffusion model for Lode Runner (as long as the file do not exist):
```
LR-conditional.bat
```

Batch file that fully trains and runs a wgan model for Lode Runner (as long as the file do not exist):
```
LR-train-wgan.bat
```
