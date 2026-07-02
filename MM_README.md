# Mega Man Generation

Generate Mega Man level scenes with a diffusion model conditioned on text input.
This Mega Man data is still experimental and on-going and the current results are not as good as the Mario levels and outputs. This mostly has to do with a smaller, more complex dataset, as well as incomplete code. Many features present in other games have not yet been implemented, but the core of the training and level generation works as intended.

## Set up the repository

```
git clone https://github.com/schrum2/MarioDiffusion.git
cd MarioDiffusion
pip install -r requirements.txt
```

## Automatic Single-Level Download

Download one Mega Man Maker level by ID.
 
```bash
cd MM_Batch
Auto_Upload_MMaker.bat 544895
```
 
Replace `544895` with the level ID you want. Saves automatically to `datasets\MM_Maker_Levels_Raw` — no prompt needed.

use this command for prompt: 

```bash
cd MM_Batch
Auto_Upload_MMaker.bat
```

Enter level ID when prompted.

## Automatic Bulk Download

Download Mega Man Maker levels by ID. The downloader begins at level ID `200000` by default (recommended to start with 100 levels).

```bash
cd megaman
python Bulk_Download.py --target 100
```

Downloaded `.mmlv` files are saved directly into:
```text
%LOCALAPPDATA%\MegaMaker\Levels
```

## Bulk Converter (MMLV --> VGLC)

Convert downloaded `.mmlv` files into VGLC `.txt` files.

```bash
cd megaman
python bulk_mmlv_to_vglc.py --output ..\datasets\MMLV_Maker_Levels
```

Saves `.txt` files into:
```text
datasets\MM_Maker_Levels
```

## Create a filtered dataset (with quality filters and source tracking)

`--stride_x` and `--stride_y` control how far the scan window moves between samples; set both to the screen size (16/14) for non-overlapping, screen-aligned extraction. `--scan_mode snap` extracts scenes that snap to fully null-free screens. `--max_enemies N` drops scenes with more than `N` enemy tiles. `--min_content_pct P` drops scenes where less than `P`% of tiles are real content. `--include_moving_ground` includes moving-ground/platform tiles, which are excluded by default since their motion isn't represented in the static tileset graphics but are needed here for full structure coverage. `--direction_captions` attaches entrance/exit direction data to each scene (required for ceiling captions to ever be generated).

All of this — filtering, captioning, tokenizing, random test caption generation, and train/validate/test splitting — can be done at once with:
```bash
cd MM_Batch
MMLV-data.bat
```

Or run each step manually:

```bash
cd ..
python megaman\bulk_mmlv_to_vglc.py --output datasets\MMLV_Maker_Levels
python create_megaman_json_data.py --levels datasets\MMLV_Maker_Levels --tileset datasets\MM.json --stride_x 16 --stride_y 14 --scan_mode snap --max_enemies 4 --min_content_pct 15 --direction_captions --include_moving_ground --output datasets\MMLV_Levels.json
python MM_create_ascii_captions.py --dataset datasets\MMLV_Levels.json --tileset datasets\MM.json --output datasets\MMLV_LevelsAndCaptions-regular.json
python tokenizer.py save --json datasets\MMLV_LevelsAndCaptions-regular.json --pkl_file datasets\MMLV_Tokenizer-regular.pkl
python create_random_test_captions.py --save_file datasets\MMLV_RandomTest-regular.json --json datasets\MMLV_LevelsAndCaptions-regular.json --seed 0 --game MM-Full
python split_data.py --json_file datasets\MMLV_LevelsAndCaptions-regular.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MM-Full
```

Train the text encoder (MLM) on the training split:
```bash
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MMLV_LevelsAndCaptions-regular-train.json --val_json datasets\MMLV_LevelsAndCaptions-regular-validate.json --test_json datasets\MMLV_LevelsAndCaptions-regular-test.json --pkl datasets\MMLV_Tokenizer-regular.pkl --output_dir MMLV-MLM-regular0 --seed 0
```

Train the text-conditional diffusion model:
```bash
python train_diffusion.py --pkl datasets\MMLV_Tokenizer-regular.pkl --json datasets\MMLV_LevelsAndCaptions-regular-train.json --val_json datasets\MMLV_LevelsAndCaptions-regular-validate.json --augment --mlm_model_dir MMLV-MLM-regular0 --text_conditional --output_dir MMLV_conditional_regular0 --seed 0 --game MM-Full
```

For a quick test run instead of waiting on full training, add `--num_epochs 2 --save_image_epochs 100000`.

**Known limitation:** Moving-ground platforms (the `M` tile) are included here via `--include_moving_ground`, but their motion is not represented in the static scene graphics. See the GitHub issue tracking proper tileset/graphics support for moving-ground platforms for the planned fix.

## Generate Levels

Interactive GUI:
```bash
python interactive_tile_level_generator.py --model_path MMLV_conditional_regular0 --load_data datasets\MMLV_LevelsAndCaptions-regular.json --game MM-Full
```

Text prompt generation:
```bash
python text_to_level_diffusion.py --model_path MMLV_conditional_regular0 --game MM-Full
```

Batch generation:
```bash
python run_diffusion.py --model_path MMLV_conditional_regular0 --num_samples 100 --text_conditional --save_as_json --output_dir MMLV_conditional_regular0-samples --level_width 16 --game MM-Full
```

Browse generated levels:
```bash
python ascii_data_browser.py MMLV_conditional_regular0-samples\all_levels.json datasets\MM.json
```

## Mega Man Maker Conversion

Generated `.txt` files can be converted back into playable Mega Man Maker levels, and downloaded `.mmlv` files can be converted into VGLC `.txt` files.

```bash
cd MM_Batch
MegaManMaker.bat path\to\file.mmlv
```
or
```bash
cd MM_Batch
MegaManMaker.bat path\to\file.txt
```

Replace the path with the file you want to convert. `.mmlv` files convert to `.txt`; `.txt` files convert to `.mmlv` and are copied automatically into your Mega Man Maker `My Levels` folder.

You can also drag and drop a file onto the window, or run it with no argument for a prompt:
```bash
cd MM_Batch
MegaManMaker.bat
```
---

## Alternate workflow: full dataset from TheVGLC

You will need to check out [my forked copy of TheVGLC](https://github.com/schrum2/TheVGLC), cloned next to `MarioDiffusion`:
```
cd ..
git clone https://github.com/schrum2/TheVGLC.git
cd MarioDiffusion
```

Create the raw 16x16 level samples, captions, and tokenizers for both sub-games:
```
python create_megaman_json_data.py --output datasets\MM_Levels-full.json
python create_megaman_json_data.py --output datasets\MM_Levels-simple.json --group_encodings

python MM_create_ascii_captions.py --dataset datasets\MM_Levels-full.json --tileset datasets\MM.json --output datasets\MM_LevelsAndCaptions-full-regular.json
python MM_create_ascii_captions.py --dataset datasets\MM_Levels-simple.json --tileset datasets\MM-simple-tileset.json --output datasets\MM_LevelsAndCaptions-simple-regular.json

python tokenizer.py save --json datasets\MM_LevelsAndCaptions-full-regular.json --pkl_file datasets\MM_Tokenizer-full-regular.pkl
python tokenizer.py save --json datasets\MM_LevelsAndCaptions-simple-regular.json --pkl_file datasets\MM_Tokenizer-simple-regular.pkl
```

All of this can be done with this batch file:
```
cd MM_Batch
MM-data.bat
```

Browse level scenes and their captions:
```
python ascii_data_browser.py datasets\MM_LevelsAndCaptions-full-regular.json datasets\MM.json
```

Train an unconditional diffusion model without any text embeddings:
```
python train_diffusion.py --json datasets\MM_LevelsAndCaptions-simple-regular.json --augment --output_dir MM_unconditional_simple0 --seed 0 --game MM-Simple
```

## Train text encoder

```
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-simple-regular.json --pkl datasets\MM_Tokenizer-simple-regular.pkl --output_dir MM-MLM-simple-regular --seed 0
```

## Train text-conditional diffusion model

```
python train_diffusion.py --pkl datasets\MM_Tokenizer-simple-regular.pkl --json datasets\MM_LevelsAndCaptions-simple-regular.json --augment --mlm_model_dir MM-MLM-simple-regular --text_conditional --output_dir MM_conditional_simple_regular0 --seed 0 --game MM-Simple
```

Speed trick — set `--save_image_epochs` larger than your epoch count to skip intermediate sample images:
```
python train_diffusion.py --pkl datasets\MM_Tokenizer-simple-regular.pkl --json datasets\MM_LevelsAndCaptions-simple-regular.json --augment --mlm_model_dir MM-MLM-simple-regular --text_conditional --output_dir MM_conditional_simple_regular0 --seed 0 --game MM-Simple --save_image_epochs 100000
```

This whole process (Simple version only) can be done with:
```
cd MM_Batch
MM_conditional.bat
```

## Generate levels in batch with run_diffusion.py

Unconditional model:
```
python run_diffusion.py --model_path MM_unconditional_simple0 --num_samples 100 --save_as_json --output_dir MM_unconditional_simple0-samples --level_width 16 --game MM-Simple
```

Text-conditional model:
```
python run_diffusion.py --model_path MM_conditional_simple_regular0 --num_samples 100 --text_conditional --save_as_json --output_dir MM_conditional_simple_regular0-samples --level_width 16 --game MM-Simple
```

Browse:
```
python ascii_data_browser.py MM_conditional_simple_regular0-samples\all_levels.json datasets\MM-simple-tileset.json
```

For the full tileset, swap `MM-Simple` with `MM-Full` and point to the appropriate model and tileset.

## Train and generate levels with block2vec tile embeddings (experimental)

By default, unconditional diffusion models represent each tile as a one-hot vector. Block2Vec replaces this with learned embedding vectors trained on 3x3 tile windows, so contextually similar tiles end up with similar vectors.

```
MM_Batch\MM_unconditional-embedding.bat {embedding_dims}
```

(`embedding_dims` is optional, default 16.)

Manual steps:

Slice the VGLC levels into 3x3 tile windows for embedding training:
```
python create_tile_level_json_data.py --tileset datasets\MM-simple-tileset.json --levels ..\TheVGLC\MegaMan\Enhanced --output datasets\MM_3x3_Tiles-simple.json --tile_size 3 --char_map datasets\MM-VGLC-to-simple.json
```

Train the block2vec embedding model on those windows:
```
python train_block2vec.py --json_file datasets\MM_3x3_Tiles-simple.json --output_dir MM-simple-block2vec%EMBEDDING_DIM%-embeddings --embedding_dim %EMBEDDING_DIM% --epochs 300
```

Train the unconditional diffusion model using the learned embeddings instead of one-hot tiles:
```
python train_diffusion.py --game MM-Simple --augment --block_embedding_model_path MM-simple-block2vec%EMBEDDING_DIM%-embeddings --output_dir MM-simple-conditional0-block2vec%EMBEDDING_DIM% --num_epochs 500 --json datasets\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\MM_LevelsAndCaptions-simple-regular-validate.json --seed 0
```

Generate levels from the trained block2vec model:
```
python run_diffusion.py --model_path MM-simple-conditional0-block2vec%EMBEDDING_DIM% --num_samples 100 --save_as_json --output_dir MM-simple-conditional0-block2vec%EMBEDDING_DIM%-samples --game MM-Simple
```

Train a conditional model with block2vec tile embeddings:
```
cd MM_Batch
MM_conditional-embeddings.bat {embedding_dims}
```

## Manual steps for more control:

Create the raw level samples (grouped/simple encoding):
```
python create_megaman_json_data.py --output datasets\MM_Levels-simple.json --group_encodings
```

Generate captions for those samples:
```
python MM_create_ascii_captions.py --dataset datasets\MM_Levels-simple.json --tileset datasets\MM-simple-tileset.json --output datasets\MM_LevelsAndCaptions-simple-regular.json
```

Build the tokenizer:
```
python tokenizer.py save --json datasets\MM_LevelsAndCaptions-simple-regular.json --pkl_file datasets\MM_Tokenizer-simple-regular.pkl
```

Create a set of random test captions for evaluation:
```
python create_random_test_captions.py --save_file datasets\MM_RandomTest_simple-regular.json --json datasets\MM_LevelsAndCaptions-simple-regular.json --seed 0 --game MM-Simple
```

Split the dataset into train/validate/test sets:
```
python split_data.py --json_file datasets\MM_LevelsAndCaptions-simple-regular.json --train_pct .9 --val_pct .05 --test_pct .05 --seed 0 --game MM-Simple
```

Train the text encoder (MLM) on the full dataset:
```
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-simple-regular.json --pkl datasets\MM_Tokenizer-simple-regular.pkl --output_dir MM-MLM-simple0 --seed 0
```

Slice the VGLC levels into 3x3 tile windows for embedding training:
```
python create_tile_level_json_data.py --tileset datasets\MM-simple-tileset.json --levels ..\TheVGLC\MegaMan\Enhanced --output datasets\MM_3x3_Tiles-simple.json --tile_size 3 --char_map datasets\MM-VGLC-to-simple.json
```

Train the block2vec embedding model on those windows:
```
python train_block2vec.py --json_file datasets\MM_3x3_Tiles-simple.json --output_dir MM-simple-block2vec%EMBEDDING_DIM%-embeddings --embedding_dim %EMBEDDING_DIM% --epochs 300
```

Train the text-conditional diffusion model on the train/validate split, using both the text encoder and the block2vec embeddings:
```
python train_diffusion.py --text_conditional --mlm_model_dir MM-MLM-simple0 --game MM-Simple --augment --block_embedding_model_path MM-simple-block2vec%EMBEDDING_DIM%-embeddings --output_dir MM-simple-conditional0-block2vec%EMBEDDING_DIM% --num_epochs 500 --json datasets\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\MM_LevelsAndCaptions-simple-regular-validate.json --seed 0
```

## Mega Man Maker

[Mega Man Maker](https://github.com/schrum2/MarioDiffusion/tree/dev_alaaAlmzayen/megaman)


## Evaluate caption adherence of text-conditional diffusion model

You can evaluate the final model's ability to adhere to input captions with this command:

```
python evaluate_caption_adherence.py --model_path MM_conditional_full_regular0 --save_as_json --json datasets\MMLV_LevelsAndCaptions-regular.json --output_dir text-to-level-final --game MM-Full
```
You can also evaluate how caption adherence changed during training with respect to the testing set:

```
python evaluate_caption_adherence.py --model_path MM_conditional_full_regular0 --save_as_json --json datasets\MMLV_LevelsAndCaptions-regular-test.json --compare_checkpoints --game MM-Full
```
However, it is easy to match captions that are similar to real game captions. You can evaluate how caption adherence changed during training with respect to previously unseen randomly generated captions too:

```
python evaluate_caption_adherence.py --model_path MM_conditional_full_regular0 --save_as_json --json datasets\MMLV_RandomTest-regular.json --compare_checkpoints --game MM-Full
```

Entrance/exit direction captions (added via `--direction_captions` earlier in the pipeline) are automatically excluded from caption-adherence scoring, so no extra flags are needed here to account for them.

If you'd like to create all the generated data used to evaluate caption adherence in one step, you can do so by running the batch file like this:

```
batch\evaluate_caption_adherence_multi.bat MM_conditional_full_regular0 regular MMLV
```