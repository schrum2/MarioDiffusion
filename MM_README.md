# Mega Man Generation

Generate Mega Man level scenes with a diffusion model conditioned on text input.
This Mega Man data is still experimental and on-going and the current results are not as good as the Mario levels and outputs. This mostly has to do with a smaller, more complex dataset, as well as incomplete code. Many features present in other games have not yet been implemented, but the core of the training and level generation works as intended.

## Set up the repository
This repository can be checked out with this command:
```
git clone https://github.com/schrum2/MarioDiffusion.git
```
Data used for training our models already exists in the `datasets` directory of this repo,
but you can recreate the data using these commands. First, you will need to check out 
[my forked copy of TheVGLC](https://github.com/schrum2/TheVGLC). Note that the following
command should be executed in the parent directory of the `MarioDiffusion` repository so that
the directories for `MarioDiffusion` and `TheVGLC` are next to each other in the same directory:
```
git clone https://github.com/schrum2/TheVGLC.git
```

Then, enter the "MarioDiffusion" repository
```
cd MarioDiffusion
```

Before running any code, install all requirements with pip:
```
pip install -r requirements.txt
```
Before being able to generate Mega Man levels, you must create a dataset which happens below.

## Create datasets

Due to the massivly increased number of tiles in Mega Man, we split our data into 2 different games internally. "MM-Full" contains the full dataset of tiles, including unique enemies and powerups, while "MM-Simple" groups things like enemies, poweups, and hazards together, giving us a boost in performance, at the cost of some complexity.

In order to create the datasets for both versions of Mega Man, we will be running all of these commands twice. First, we need to create the raw 16X16 level samples with these commands:
```
python create_megaman_json_data.py --output datasets\\MM_Levels_Full.json
python create_megaman_json_data.py --output datasets\\MM_Levels_Simple.json --group_encodings
```

The next step is to create captions for these raw levels, which can be done with this command:
```
python MM_create_ascii_captions.py --dataset datasets\\MM_Levels_Full.json --tileset datasets\\MM.json --output datasets\\MM_LevelsAndCaptions-full-regular.json
python MM_create_ascii_captions.py --dataset datasets\\MM_Levels_Simple.json --tileset datasets\\MM_Simple_Tileset.json --output datasets\\MM_LevelsAndCaptions-simple-regular.json
```
The last step is to create tokenizers for our data, which can be done like this:
```
python tokenizer.py save --json datasets\\MM_LevelsAndCaptions-full-regular.json --pkl_file datasets\MM_Tokenizer-full-regular.pkl
python tokenizer.py save --json datasets\\MM_LevelsAndCaptions-simple-regular.json --pkl_file datasets\MM_Tokenizer-simple-regular.pkl
```

All of this can be done with this batch file, which runs each of these commands in sequence

```
cd MM_Batch
MM-data.bat
```
Now you can browse level scenes and their captions with a command like this (the json file can be replaced by any levels and captions json file in datasets):
```
python ascii_data_browser.py datasets\MM_LevelsAndCaptions-full-regular.json datasets\MM.json
```


## Train unconditional diffusion model

To train an unconditional diffusion model without any text embeddings, run this command:
```
python train_diffusion.py --json datasets\\MM_LevelsAndCaptions-simple-regular.json --augment --output_dir MM_unconditional_simple0 --seed 0 --game MM-Simple
```

## Train text encoder

Masked language modeling is used to train the text embedding model. Use any dataset with an appropriate tokenizer, we will default to the ones for MM-Simple for the rest of the commands here, though both sub-games work fine.
```
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-simple-regular.json --pkl datasets\MM_Tokenizer-simple-regular.pkl --output_dir MM-MLM-simple-regular --seed 0
```

## Train text-conditional diffusion model

Now that the text embedding model is ready, train a diffusion model conditioned on text embeddings from the descriptive captions. Note that this can take a while. We used relatively modest consumer GPUs, so our models took about 12 hours to train:
```
python train_diffusion.py --pkl datasets\MM_Tokenizer-simple-regular.pkl --json datasets\\MM_LevelsAndCaptions-simple-regular.json --augment --mlm_model_dir MM-MLM-simple-regular --text_conditional --output_dir MM_conditional_simple_regular0 --seed 0 --game MM-Simple
```
Another trick if you care more about speed than seeing intermediate results is to set `--save_image_epochs` to a large number (larger than the number of epochs), like this
```
python train_diffusion.py --pkl datasets\MM_Tokenizer-simple-regular.pkl --json datasets\\MM_LevelsAndCaptions-simple-regular.json --augment --mlm_model_dir MM-MLM-simple-regular --text_conditional --output_dir MM_conditional_simple_regular0 --seed 0 --game MM-Simple --save_image_epochs 100000
```

This process, from creating the level sample files all the way to diffusion training, can be done with this batch file (This only trains and runs the Simple version):
```
cd MM_Batch
MM_conditional.bat
```


## Generate levels from text-conditional diffusion model

In order to generate levels from a base caption, use this command
```
python text_to_level_diffusion.py --model_path MM_conditional_simple_regular0 --game MM-Simple
```
An easier-to-use GUI interface will let you select and combine known caption phrases to send to the model. Note that the selection of known phrases needs to come from the dataset you trained on.
```
python interactive_tile_level_generator.py --model_path MM_conditional_simple_regular0 --load_data datasets\\MM_LevelsAndCaptions-simple-regular.json --game MM-Simple
```