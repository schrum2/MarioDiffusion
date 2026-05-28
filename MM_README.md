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

Due to the massively increased number of tiles in Mega Man, we split our data into 2 different games internally. "MM-Full" contains the full dataset of tiles, including unique enemies and powerups, while "MM-Simple" groups things like enemies, powerups, and hazards together, giving us a boost in performance, at the cost of some complexity.

In order to create the datasets for both versions of Mega Man, run all of these commands. First, create the raw 16x16 level samples:
```
python create_megaman_json_data.py --output datasets\\MM_Levels_Full.json
python create_megaman_json_data.py --output datasets\\MM_Levels_Simple.json --group_encodings
```

Next, create captions for these raw levels:
```
python MM_create_ascii_captions.py --dataset datasets\\MM_Levels_Full.json --tileset datasets\\MM.json --output datasets\\MM_LevelsAndCaptions-full-regular.json
python MM_create_ascii_captions.py --dataset datasets\\MM_Levels_Simple.json --tileset datasets\\MM_Simple_Tileset.json --output datasets\\MM_LevelsAndCaptions-simple-regular.json
```

Create tokenizers for the data:
```
python tokenizer.py save --json datasets\\MM_LevelsAndCaptions-full-regular.json --pkl_file datasets\MM_Tokenizer-full-regular.pkl
python tokenizer.py save --json datasets\\MM_LevelsAndCaptions-simple-regular.json --pkl_file datasets\MM_Tokenizer-simple-regular.pkl
```

Create random test captions for MM-Simple (sampled from the dataset, since MM does not yet have a grammar-based caption generator):
```
python create_random_test_captions.py --save_file "datasets\\MM_RandomTest-simple-regular.json" --json datasets\\MM_LevelsAndCaptions-simple-regular.json --seed 0 --game MM-Simple
```

Split the captioned datasets into train, validation, and test sets:
```
python split_data.py --json datasets\\MM_LevelsAndCaptions-simple-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game megaman
python split_data.py --json datasets\\MM_LevelsAndCaptions-full-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 0 --game megaman
```

All of the above steps can be run at once with this batch file:
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
python train_diffusion.py --json datasets\\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\\MM_LevelsAndCaptions-simple-regular-validate.json --augment --output_dir MM_unconditional_simple0 --seed 0 --game MM-Simple
```

## Train text encoder

Masked language modeling is used to train the text embedding model. Use any dataset with an appropriate tokenizer. We default to MM-Simple for the rest of the commands here, though both sub-games work fine.
```
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\\MM_LevelsAndCaptions-simple-regular-validate.json --test_json datasets\\MM_LevelsAndCaptions-simple-regular-test.json --pkl datasets\\MM_Tokenizer-simple-regular.pkl --output_dir MM-MLM-simple-regular --seed 0
```

## Train text-conditional diffusion model

Now that the text embedding model is ready, train a diffusion model conditioned on text embeddings from the descriptive captions. Note that this can take a while. We used relatively modest consumer GPUs, so our models took about 12 hours to train:
```
python train_diffusion.py --pkl datasets\\MM_Tokenizer-simple-regular.pkl --json datasets\\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\\MM_LevelsAndCaptions-simple-regular-validate.json --augment --mlm_model_dir MM-MLM-simple-regular --text_conditional --output_dir MM_conditional_simple_regular0 --seed 0 --game MM-Simple --plot_validation_caption_score
```

If you care more about speed than seeing intermediate results, set `--save_image_epochs` to a large number (larger than the number of epochs):
```
python train_diffusion.py --pkl datasets\\MM_Tokenizer-simple-regular.pkl --json datasets\\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\\MM_LevelsAndCaptions-simple-regular-validate.json --augment --mlm_model_dir MM-MLM-simple-regular --text_conditional --output_dir MM_conditional_simple_regular0 --seed 0 --game MM-Simple --plot_validation_caption_score --save_image_epochs 100000
```

This entire process, from creating the level sample files all the way to diffusion training, can be done with this batch file (trains and runs the Simple version only):
```
cd MM_Batch
MM-conditional.bat
```


## Generate levels from text-conditional diffusion model

To generate levels from a text caption, use this command:
```
python text_to_level_diffusion.py --model_path MM_conditional_simple_regular0 --game MM-Simple
```

An easier-to-use GUI interface lets you select and combine known caption phrases to send to the model. Note that the selection of known phrases needs to come from the dataset you trained on:
```
python interactive_tile_level_generator.py --model_path MM_conditional_simple_regular0 --load_data datasets\\MM_LevelsAndCaptions-simple-regular.json --game MM-Simple
```

## Evaluate caption adherence of text-conditional diffusion model

You can evaluate the final model's ability to follow input captions with this command:
```
python evaluate_caption_adherence.py --model_path MM_conditional_simple_regular0 --save_as_json --json datasets\\MM_LevelsAndCaptions-simple-regular.json --output_dir MM-text-to-level-eval
```

You can also evaluate how caption adherence changed during training with respect to the test set:
```
python evaluate_caption_adherence.py --model_path MM_conditional_simple_regular0 --save_as_json --json datasets\\MM_LevelsAndCaptions-simple-regular-test.json --compare_checkpoints
```

To evaluate against the randomly sampled held-out test captions:
```
python evaluate_caption_adherence.py --model_path MM_conditional_simple_regular0 --save_as_json --json datasets\\MM_RandomTest-simple-regular.json --compare_checkpoints
```