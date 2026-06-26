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
python create_megaman_json_data.py --output datasets\\MM_Levels-full.json
python create_megaman_json_data.py --output datasets\\MM_Levels-simple.json --group_encodings
```

The next step is to create captions for these raw levels, which can be done with this command:
```
python MM_create_ascii_captions.py --dataset datasets\\MM_Levels-full.json --tileset datasets\\MM.json --output datasets\\MM_LevelsAndCaptions-full-regular.json
python MM_create_ascii_captions.py --dataset datasets\\MM_Levels-simple.json --tileset datasets\\MM-simple-tileset.json --output datasets\\MM_LevelsAndCaptions-simple-regular.json
```
The last step is to create tokenizers for our data, which can be done like this:

```
python tokenizer.py save --json datasets\\MM_LevelsAndCaptions-full-regular.json --pkl_file datasets\MM_Tokenizer-full-regular.pkl
python tokenizer.py save --json datasets\\MM_LevelsAndCaptions-simple-regular.json --pkl_file datasets\MM_Tokenizer-simple-regular.pkl
```

All of this can be done with this batch file, which runs each of these commands in sequence:
```
cd MM_Batch
MM-data.bat
```

Now you can browse level scenes and their captions with a command like this (the json file can be replaced by any levels and captions json file in datasets):
```
python ascii_data_browser.py datasets\MM_LevelsAndCaptions-full-regular.json datasets\MM.json
```

To train an unconditional diffusion model without any text embeddings, run this command:
```
python train_diffusion.py --json datasets\\MM_LevelsAndCaptions-simple-regular.json --augment --output_dir MM_unconditional_simple0 --seed 0 --game MM-Simple
```

## Train text encoder

Masked language modeling is used to train the text embedding model. Use any dataset with an appropriate tokenizer, we will default to the ones for MM-Simple for the rest of the commands here, though both sub-games work fine:
```
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-simple-regular.json --pkl datasets\MM_Tokenizer-simple-regular.pkl --output_dir MM-MLM-simple-regular --seed 0
```

## Train text-conditional diffusion model

Now that the text embedding model is ready, train a diffusion model conditioned on text embeddings from the descriptive captions. Note that this can take a while. We used relatively modest consumer GPUs, so our models took about 12 hours to train:
```
python train_diffusion.py --pkl datasets\MM_Tokenizer-simple-regular.pkl --json datasets\\MM_LevelsAndCaptions-simple-regular.json --augment --mlm_model_dir MM-MLM-simple-regular --text_conditional --output_dir MM_conditional_simple_regular0 --seed 0 --game MM-Simple
```
Another trick if you care more about speed than seeing intermediate results is to set `--save_image_epochs` to a large number (larger than the number of epochs), like this:

```
python train_diffusion.py --pkl datasets\MM_Tokenizer-simple-regular.pkl --json datasets\\MM_LevelsAndCaptions-simple-regular.json --augment --mlm_model_dir MM-MLM-simple-regular --text_conditional --output_dir MM_conditional_simple_regular0 --seed 0 --game MM-Simple --save_image_epochs 100000
```

This process, from creating the level sample files all the way to diffusion training, can be done with this batch file (This only trains and runs the Simple version):
```
cd MM_Batch
MM_conditional.bat
```


## Generate levels from text-conditional diffusion model

In order to generate levels from a base caption, use this command:
```
python text_to_level_diffusion.py --model_path MM_conditional_simple_regular0 --game MM-Simple
```
An easier-to-use GUI interface will let you select and combine known caption phrases to send to the model. Note that the selection of known phrases needs to come from the dataset you trained on:

```
python interactive_tile_level_generator.py --model_path MM_conditional_simple_regular0 --load_data datasets\\MM_LevelsAndCaptions-simple-regular.json --game MM-Simple
```

## Generate levels in batch with run_diffusion.py

To generate a batch of levels from an unconditional MM-Simple model:

```
python run_diffusion.py --model_path MM_unconditional_simple0 --num_samples 100 --save_as_json --output_dir MM_unconditional_simple0-samples --level_width 16 -```
-game MM-Simple
```
For a text-conditional model, add `--text_conditional`:

```
python run_diffusion.py --model_path MM_conditional_simple_regular0 --num_samples 100 --text_conditional --save_as_json --output_dir MM_conditional_simple_regular0-samples --level_width 16 --game MM-Simple
```
Browse the generated levels with:

```
python ascii_data_browser.py MM_conditional_simple_regular0-samples\all_levels.json datasets\MM-simple-tileset.json
```
For the full tileset, swap `MM-Simple` with `MM-Full` and point to the appropriate model and tileset.

## Mega Man Maker

This is the link to learn more about Mega Man Maker:

[Mega Man Maker](https://github.com/schrum2/MarioDiffusion/tree/dev_alaaAlmzayen/megaman)


## Train and generate levels from unconditional model with block2vec tile embedding model (experimental)

By default, unconditional diffusion models represent each tile as a one-hot vector. Block2Vec replaces this representation with learned embedding vectors for each tile type. It is trained on 3×3 tile windows so that tiles that are contextually similar in the game end up with similar vectors. 

To train and run an unconditional model with tile embeddings, you can run this batch file
and opt to include an argument for the size of the latent embedding space by including an integer for the number of embedding dimensions (default 16)
```
MM_batch\MM_unconditional-embedding.bat {embedding_dims}
```

You can gain more control in the process and train a tile embedding model from 3x3 tile samples:
``` 
python create_tile_level_json_data.py --tileset datasets\MM-simple-tileset.json --levels ..\TheVGLC\MegaMan\Enhanced --output datasets\MM_3x3_Tiles-simple.json --tile_size 3 --char_map datasets\MM-VGLC-to-simple.json

python train_block2vec.py --json_file datasets\MM_3x3_Tiles-simple.json --output_dir MM-simple-block2vec%EMBEDDING_DIM%-embeddings --embedding_dim %EMBEDDING_DIM% --epochs 300
```
Training diffusion model with block2vec tile embeddings instead of one-hot encoding
``` 
python train_diffusion.py  --game MM-Simple --augment --block_embedding_model_path MM-simple-block2vec%EMBEDDING_DIM%-embeddings --output_dir MM-simple-conditional0-block2vec%EMBEDDING_DIM% --num_epochs 500 --json datasets\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\MM_LevelsAndCaptions-simple-regular-validate.json --seed 0 
```
Generating levels
``` 
python run_diffusion.py --model_path --output_dir MM-simple-conditional0-block2vec%EMBEDDING_DIM% --num_samples 100 --save_as_json --output_dir "Mar1and2-unconditional-block2vec-samples" --game MM-Simple 
```

##

Additionally, you can train a conditional model with block2vec tile embeddings by running this batch file:
```
cd batch
MM_conditional-embeddings.bat {embedding dims}
```
If you would like more control in the process, you can follow these steps:
```

python create_megaman_json_data.py --output datasets\MM_Levels-simple.json --group_encodings

python MM_create_ascii_captions.py --dataset datasets\MM_Levels-simple.json --tileset datasets\MM-simple-tileset.json --output datasets\MM_LevelsAndCaptions-simple-regular.json

python tokenizer.py save --json_file datasets\MM_LevelsAndCaptions-simple-regular.json --pkl_file datasets\MM_Tokenizer-simple-regular.pkl

python create_random_test_captions.py --save_file datasets\MM_RandomTest_simple-regular.json --json datasets\MM_LevelsAndCaptions-simple-regular.json --seed 0 --game MM-Simple

python split_data.py --json_file datasets\MM_LevelsAndCaptions-simple-regular.json --train_pct .9 --val_pct .05 --test_pct .05 --seed 0 --game mm-simple

python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-simple-regular.json --pkl datasets\MM_Tokenizer-simple-regular.pkl --output_dir MM-MLM-simple0 --seed 0

python create_tile_level_json_data.py --tileset datasets\MM-simple-tileset.json --levels ..\TheVGLC\MegaMan\Enhanced --output datasets\MM_3x3_Tiles-simple.json --tile_size 3 --char_map datasets\MM-VGLC-to-simple.json

python train_block2vec.py --json_file datasets\MM_3x3_Tiles-simple.json --output_dir MM-simple-block2vec%EMBEDDING_DIM%-embeddings --embedding_dim %EMBEDDING_DIM% --epochs 300

python train_diffusion.py --text_conditional --mlm_model_dir MM-MLM-simple0 --game MM-Simple --augment --block_embedding_model_path MM-simple-block2vec%EMBEDDING_DIM%-embeddings --output_dir MM-simple-conditional0-block2vec%EMBEDDING_DIM% --num_epochs 500 --json datasets\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\MM_LevelsAndCaptions-simple-regular-validate.json --seed 0
```

## Create a filtered dataset (with quality filters and source tracking)

The following new options are available for `create_megaman_json_data.py`. `--stride_x` and `--stride_y` control how far the scan window moves between samples (sliding_window/snap modes only); set both to the screen size (e.g. 16/14) for non-overlapping, screen-aligned extraction. `--scan_mode snap` extracts wide and tall scenes that snap to fully null-free screens. `--max_enemies N` drops any scene with more than `N` enemy tiles. `--min_content_pct P` drops any scene where less than `P`% of tiles are real content (i.e. not empty/passable/null), filtering out near-empty, unplayable scenes. `--include_moving_ground` includes scenes containing moving-ground/platform tiles, which are excluded by default since their motion isn't represented in the static tileset graphics.


Generate a filtered, screen-aligned dataset with this command:
```
python create_megaman_json_data.py --levels ..\TheVGLC\MegaMan\Enhanced --stride_x 16 --stride_y 14 --scan_mode snap --max_enemies 4 --min_content_pct 15 --output datasets\MM_Levels_Filtered.json
```
Then generate deterministic captions for it:
```
python MM_create_ascii_captions.py --dataset datasets\MM_Levels_Filtered.json --tileset datasets\MM.json --output datasets\MM_LevelsAndCaptions-filtered-regular.json --describe_absence
```
Build a tokenizer:
```
python tokenizer.py save --json datasets\MM_LevelsAndCaptions-filtered-regular.json --pkl_file datasets\MM_Tokenizer-filtered-regular.pkl
```
Train the text encoder (MLM):
```
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-filtered-regular.json --pkl datasets\MM_Tokenizer-filtered-regular.pkl --output_dir MM-MLM-filtered-regular --seed 0
```
Train the text-conditional diffusion model:
```
python train_diffusion.py --pkl datasets\MM_Tokenizer-filtered-regular.pkl --json datasets\MM_LevelsAndCaptions-filtered-regular.json --augment --mlm_model_dir MM-MLM-filtered-regular --text_conditional --output_dir MM_conditional_filtered_regular0 --seed 0 --game MM-Full
```
Use `--game MM-Simple` instead if the dataset was generated with `--group_encodings`.

**Known limitation:** Moving-ground platforms (the `M` tile) are excluded by default rather than properly represented, since we don't yet have graphics or a static-scene encoding for their motion. See the GitHub issue tracking proper tileset/graphics support for moving-ground platforms for the planned fix.