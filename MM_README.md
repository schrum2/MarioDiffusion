# Mega Man Generation

Generate Mega Man level scenes with a diffusion model conditioned on text input.

This Mega Man data is still experimental and ongoing. Current results are not as strong as Mario generation due to a smaller and more complex dataset, as well as incomplete code support. Many features present in other games have not yet been implemented, but the core training and generation pipeline works as intended.

---

# Repository Setup

Clone the repository:

```bash
git clone https://github.com/schrum2/MarioDiffusion.git
```

Enter repository:

```bash
cd MarioDiffusion
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Some older dataset pipelines require TheVGLC repository.

Clone in the parent directory of MarioDiffusion:

```bash
git clone https://github.com/schrum2/TheVGLC.git
```

Directory structure should look like:

```text
ParentFolder/
 ├── MarioDiffusion/
 └── TheVGLC/
```

---

# Mega Man Maker Full Pipeline (Online Levels → Diffusion Model)

This pipeline downloads Mega Man Maker levels directly from online, converts them to VGLC format, creates datasets, captions them, and trains a text-conditional diffusion model.

---

## Step 1: Download Levels in Bulk

Download Mega Man Maker levels by ID, The downloader begins at level ID `200000` by default.

Recommended starting dataset:

```text
100 levels
```

Run:

```bash
cd megaman
python Bulk_Download.py --target 100
```

Downloaded `.mmlv` files are saved directly into:

```text
%USERPROFILE%\AppData\Local\MegaMaker\Levels
```

---

## Step 2: Convert Levels to ASCII (VGLC Format)

Convert downloaded `.mmlv` files into VGLC `.txt` files.

Run:

```bash
cd megaman
python bulk_mmlv_to_vglc.py --output ..\datasets\MM_Maker_Levels
```

- Converts every file to ASCII format
- Saves `.txt` files into:

```text
datasets\MM_Maker_Levels
```

---


## Step 3: Create Filtered Dataset

Create a filtered dataset from converted levels.

The following options are available.

`--stride_x`

Horizontal scan distance.

`--stride_y`

Vertical scan distance.

`--scan_mode snap`

Extract screen-aligned scenes.

`--max_enemies N`

Remove scenes with too many enemies.

`--min_content_pct P`

Remove nearly empty scenes.

`--include_moving_ground`

Include moving platform tiles.

Generate filtered dataset:

```bash
cd ..
python create_megaman_json_data.py --levels datasets\MM_Maker_Levels --tileset datasets\MM.json --stride_x 16 --stride_y 14 --scan_mode snap --max_enemies 4 --min_content_pct 15 --output datasets\MM_Levels_Filtered.json
```

---

## Step 4: Generate Captions

Generate deterministic captions.

Run:

```bash
python MM_create_ascii_captions.py --dataset datasets\MM_Levels_Filtered.json --tileset datasets\MM.json --output datasets\MM_LevelsAndCaptions-filtered-regular.json
```

---

## Step 5: Build Tokenizer

```bash
python tokenizer.py save --json datasets\MM_LevelsAndCaptions-filtered-regular.json --pkl_file datasets\MM_Tokenizer-filtered-regular.pkl
```

---

## Step 6: Train MLM Text Encoder

```bash
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-filtered-regular.json --pkl datasets\MM_Tokenizer-filtered-regular.pkl --output_dir MM-MLM-filtered-regular --seed 0
```

---

## Step 7: Train Conditional Diffusion Model

```bash
python train_diffusion.py --pkl datasets\MM_Tokenizer-filtered-regular.pkl --json datasets\MM_LevelsAndCaptions-filtered-regular.json --augment --mlm_model_dir MM-MLM-filtered-regular --text_conditional --output_dir MM_conditional_filtered_regular0 --seed 0 --game MM-Full
```

To disable image saving:

```bash
--save_image_epochs 100000
```

---

## Step 8: Generate Levels

Interactive GUI:

```bash
python interactive_tile_level_generator.py --model_path MM_conditional_filtered_regular0 --load_data datasets\MM_LevelsAndCaptions-filtered-regular.json --game MM-Full
```

Text prompt generation:

```bash
python text_to_level_diffusion.py --model_path MM_conditional_filtered_regular0 --game MM-Full
```

Batch generation:

```bash
python run_diffusion.py --model_path MM_conditional_filtered_regular0 --num_samples 100 --text_conditional --save_as_json --output_dir MM_conditional_filtered_regular0-samples --level_width 16 --game MM-Full
```

Browse generated levels:

```bash
python ascii_data_browser.py MM_conditional_filtered_regular0-samples\all_levels.json datasets\MM.json
```

---


# Create Datasets (Legacy Full + Simple Pipeline)

Mega Man internally supports two dataset modes.

## MM-Full

Full tile dataset.

Includes:

- all enemies
- all hazards
- all powerups
- full tile diversity

Tileset:

```text
datasets\MM.json
```

---

## MM-Simple

Compressed tile dataset.

Groups:

- enemies together
- hazards together
- powerups together

Trades complexity for better training performance.

Tileset:

```text
datasets\MM-simple-tileset.json
```

---

## Create Raw Dataset Files

Create both datasets.

MM-Full:

```bash
python create_megaman_json_data.py --output datasets\MM_Levels-full.json
```

MM-Simple:

```bash
python create_megaman_json_data.py --output datasets\MM_Levels-simple.json --group_encodings
```

# Caption Generation

After raw datasets are created, captions must be generated for each scene.

Captions are deterministic and rule-based (not LLM-generated).

---

## Generate Captions

### MM-Full

```bash
python MM_create_ascii_captions.py --dataset datasets\MM_Levels-full.json --tileset datasets\MM.json --output datasets\MM_LevelsAndCaptions-full-regular.json
```

---

### MM-Simple

```bash
python MM_create_ascii_captions.py --dataset datasets\MM_Levels-simple.json --tileset datasets\MM-simple-tileset.json --output datasets\MM_LevelsAndCaptions-simple-regular.json
```

Output files:

```text
datasets\MM_LevelsAndCaptions-full-regular.json
datasets\MM_LevelsAndCaptions-simple-regular.json
```

---

# Build Tokenizers

A tokenizer converts captions into vocabulary tokens used by the MLM and diffusion model.

---

## MM-Full Tokenizer

```bash
python tokenizer.py save --json datasets\MM_LevelsAndCaptions-full-regular.json --pkl_file datasets\MM_Tokenizer-full-regular.pkl
```

---

## MM-Simple Tokenizer

```bash
python tokenizer.py save --json datasets\MM_LevelsAndCaptions-simple-regular.json --pkl_file datasets\MM_Tokenizer-simple-regular.pkl
```

---

# Automated Dataset Pipeline Batch Script

The following batch file runs:

- dataset creation
- caption generation
- tokenizer generation

Run:

```bash
cd MM_Batch
MM-data.bat
```

This executes the preprocessing pipeline automatically.

---

# Browse Dataset Scenes

Browse extracted level scenes and their captions.

Example using MM-Full:

```bash
python ascii_data_browser.py datasets\MM_LevelsAndCaptions-full-regular.json datasets\MM.json
```

Example using MM-Simple:

```bash
python ascii_data_browser.py datasets\MM_LevelsAndCaptions-simple-regular.json datasets\MM-simple-tileset.json
```

---

# Train Unconditional Diffusion Model

Unconditional diffusion trains without text embeddings.

---

## Train MM-Simple Unconditional Model

```bash
python train_diffusion.py --json datasets\MM_LevelsAndCaptions-simple-regular.json --augment --output_dir MM_unconditional_simple0 --seed 0 --game MM-Simple
```

Output:

```text
MM_unconditional_simple0
```

This model generates levels without captions.

---

# Train Text Encoder (MLM)

Masked Language Modeling trains the text embedding model.

Any caption dataset may be used.

Examples below default to MM-Simple.

---

## Train MLM on MM-Simple

```bash
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-simple-regular.json --pkl datasets\MM_Tokenizer-simple-regular.pkl --output_dir MM-MLM-simple-regular --seed 0
```

Output:

```text
MM-MLM-simple-regular
```

---

## Train MLM on MM-Full

```bash
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-full-regular.json --pkl datasets\MM_Tokenizer-full-regular.pkl --output_dir MM-MLM-full-regular --seed 0
```

---

# Train Text-Conditional Diffusion Model

Once MLM training is complete, train a diffusion model conditioned on text captions.

Training can take:

```text
~12 hours on consumer GPU
```

---

## MM-Simple Conditional Model

```bash
python train_diffusion.py --pkl datasets\MM_Tokenizer-simple-regular.pkl --json datasets\MM_LevelsAndCaptions-simple-regular.json --augment --mlm_model_dir MM-MLM-simple-regular --text_conditional --output_dir MM_conditional_simple_regular0 --seed 0 --game MM-Simple
```

---

## MM-Full Conditional Model

```bash
python train_diffusion.py --pkl datasets\MM_Tokenizer-full-regular.pkl --json datasets\MM_LevelsAndCaptions-full-regular.json --augment --mlm_model_dir MM-MLM-full-regular --text_conditional --output_dir MM_conditional_full_regular0 --seed 0 --game MM-Full
```

---

## Faster Training Option

If intermediate images are not needed, disable frequent image saving.

Add:

```bash
--save_image_epochs 100000
```

Example:

```bash
python train_diffusion.py --pkl datasets\MM_Tokenizer-simple-regular.pkl --json datasets\MM_LevelsAndCaptions-simple-regular.json --augment --mlm_model_dir MM-MLM-simple-regular --text_conditional --output_dir MM_conditional_simple_regular0 --seed 0 --game MM-Simple --save_image_epochs 100000
```

---

# Automated Conditional Pipeline Batch Script

This batch file executes:

- dataset creation
- caption generation
- tokenizer creation
- MLM training
- conditional diffusion training

Run:

```bash
cd MM_Batch
MM_conditional.bat
```

Note:

This batch file currently trains only:

```text
MM-Simple
```

---

## Generate Filtered Captions

Corrected command:

```bash
python MM_create_ascii_captions.py --dataset datasets\MM_Levels_Filtered.json --tileset datasets\MM.json --output datasets\MM_LevelsAndCaptions-filtered-regular.json
```

Important:

Do NOT use:

```bash
--describe_absence
```

It has been removed.

---

## Build Filtered Tokenizer

```bash
python tokenizer.py save --json datasets\MM_LevelsAndCaptions-filtered-regular.json --pkl_file datasets\MM_Tokenizer-filtered-regular.pkl
```

---

## Train Filtered MLM

```bash
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-filtered-regular.json --pkl datasets\MM_Tokenizer-filtered-regular.pkl --output_dir MM-MLM-filtered-regular --seed 0
```

---

## Train Filtered Conditional Model

```bash
python train_diffusion.py --pkl datasets\MM_Tokenizer-filtered-regular.pkl --json datasets\MM_LevelsAndCaptions-filtered-regular.json --augment --mlm_model_dir MM-MLM-filtered-regular --text_conditional --output_dir MM_conditional_filtered_regular0 --seed 0 --game MM-Full
```

Use:

```bash
--game MM-Simple
```

if dataset used:

```bash
--group_encodings
```

# Generate Levels from Trained Models

After training finishes, levels can be generated from either unconditional or text-conditional models.

---

# Generate from Text Prompt

Generate levels directly from a text description.

Example using MM-Simple:

```bash
python text_to_level_diffusion.py --model_path MM_conditional_simple_regular0 --game MM-Simple
```

Example using MM-Full:

```bash
python text_to_level_diffusion.py --model_path MM_conditional_full_regular0 --game MM-Full
```

The program will prompt for text input.

Example prompt:

```text
many enemies. ladder on left. open space.
```

---

# Interactive GUI Generator

A GUI allows phrase selection from known captions.

This is often easier than manual prompt writing.

The phrase selection is built from the dataset used during training.

---

## MM-Simple GUI

```bash
python interactive_tile_level_generator.py --model_path MM_conditional_simple_regular0 --load_data datasets\MM_LevelsAndCaptions-simple-regular.json --game MM-Simple
```

---

## MM-Full GUI

```bash
python interactive_tile_level_generator.py --model_path MM_conditional_full_regular0 --load_data datasets\MM_LevelsAndCaptions-full-regular.json --game MM-Full
```

---

# Batch Generation with run_diffusion.py

Generate large batches of levels automatically.

---

## Unconditional MM-Simple Generation

```bash
python run_diffusion.py --model_path MM_unconditional_simple0 --num_samples 100 --save_as_json --output_dir MM_unconditional_simple0-samples --level_width 16 --game MM-Simple
```

---

## Conditional MM-Simple Generation

```bash
python run_diffusion.py --model_path MM_conditional_simple_regular0 --num_samples 100 --text_conditional --save_as_json --output_dir MM_conditional_simple_regular0-samples --level_width 16 --game MM-Simple
```

---

## Conditional MM-Full Generation

```bash
python run_diffusion.py --model_path MM_conditional_full_regular0 --num_samples 100 --text_conditional --save_as_json --output_dir MM_conditional_full_regular0-samples --level_width 16 --game MM-Full
```

---

# Browse Generated Levels

View generated levels visually.

MM-Simple:

```bash
python ascii_data_browser.py MM_conditional_simple_regular0-samples\all_levels.json datasets\MM-simple-tileset.json
```

MM-Full:

```bash
python ascii_data_browser.py MM_conditional_full_regular0-samples\all_levels.json datasets\MM.json
```

---

# Mega Man Maker Conversion and Playtesting

Generated `.txt` files can be converted back into playable Mega Man Maker levels.

Use:

```bash
cd MM_batch
MegaManMaker.bat
```

Drag and drop:

- `.mmlv` → converts to `.txt`
- `.txt` → converts to `.mmlv`

They appear in Mega Man Maker under:

```text
My Levels
```
---

# Automatic Single-Level Download

Download one Mega Man Maker level by ID.

Run:

```bash
cd MM_Batch
Auto_Upload_MMMaker.bat
```

Enter level ID when prompted.

---

# Unconditional Block2Vec Pipeline

Batch file:

```bash
MM_batch\MM_unconditional-embedding.bat {embedding_dims}
```

Default embedding dimension:

```text
16
```

---

## Create 3x3 Tile Dataset

```bash
python create_tile_level_json_data.py --tileset datasets\MM-simple-tileset.json --levels ..\TheVGLC\MegaMan\Enhanced --output datasets\MM_3x3_Tiles-simple.json --tile_size 3 --char_map datasets\MM-VGLC-to-simple.json
```

---

## Train Block2Vec Embeddings

```bash
python train_block2vec.py --json_file datasets\MM_3x3_Tiles-simple.json --output_dir MM-simple-block2vec%EMBEDDING_DIM%-embeddings --embedding_dim %EMBEDDING_DIM% --epochs 300
```

---

## Train Diffusion with Block Embeddings

```bash
python train_diffusion.py --game MM-Simple --augment --block_embedding_model_path MM-simple-block2vec%EMBEDDING_DIM%-embeddings --output_dir MM-simple-conditional0-block2vec%EMBEDDING_DIM% --num_epochs 500 --json datasets\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\MM_LevelsAndCaptions-simple-regular-validate.json --seed 0
```

---

## Generate Levels with Block2Vec Model

```bash
python run_diffusion.py --model_path MM-simple-conditional0-block2vec%EMBEDDING_DIM% --num_samples 100 --save_as_json --output_dir MM-simple-block2vec-samples --game MM-Simple
```

---

# Conditional Block2Vec Pipeline

Batch script:

```bash
cd MM_batch
MM_conditional-embeddings.bat {embedding_dims}
```

---

## Manual Pipeline

Create MM-Simple dataset:

```bash
python create_megaman_json_data.py --output datasets\MM_Levels-simple.json --group_encodings
```

Generate captions:

```bash
python MM_create_ascii_captions.py --dataset datasets\MM_Levels-simple.json --tileset datasets\MM-simple-tileset.json --output datasets\MM_LevelsAndCaptions-simple-regular.json
```

Create tokenizer:

```bash
python tokenizer.py save --json datasets\MM_LevelsAndCaptions-simple-regular.json --pkl_file datasets\MM_Tokenizer-simple-regular.pkl
```

Create random test captions:

```bash
python create_random_test_captions.py --save_file datasets\MM_RandomTest_simple-regular.json --json datasets\MM_LevelsAndCaptions-simple-regular.json --seed 0 --game MM-Simple
```

Split dataset:

```bash
python split_data.py --json_file datasets\MM_LevelsAndCaptions-simple-regular.json --train_pct .9 --val_pct .05 --test_pct .05 --seed 0 --game MM-Simple
```

Train MLM:

```bash
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\MM_LevelsAndCaptions-simple-regular.json --pkl datasets\MM_Tokenizer-simple-regular.pkl --output_dir MM-MLM-simple0 --seed 0
```

Create tile dataset:

```bash
python create_tile_level_json_data.py --tileset datasets\MM-simple-tileset.json --levels ..\TheVGLC\MegaMan\Enhanced --output datasets\MM_3x3_Tiles-simple.json --tile_size 3 --char_map datasets\MM-VGLC-to-simple.json
```

Train Block2Vec:

```bash
python train_block2vec.py --json_file datasets\MM_3x3_Tiles-simple.json --output_dir MM-simple-block2vec%EMBEDDING_DIM%-embeddings --embedding_dim %EMBEDDING_DIM% --epochs 300
```

Train conditional diffusion:

```bash
python train_diffusion.py --text_conditional --mlm_model_dir MM-MLM-simple0 --game MM-Simple --augment --block_embedding_model_path MM-simple-block2vec%EMBEDDING_DIM%-embeddings --output_dir MM-simple-conditional0-block2vec%EMBEDDING_DIM% --num_epochs 500 --json datasets\MM_LevelsAndCaptions-simple-regular-train.json --val_json datasets\MM_LevelsAndCaptions-simple-regular-validate.json --seed 0
```
---