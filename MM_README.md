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
python bulk_mmlv_to_vglc.py --output ..\datasets\MM_Maker_Levels
```

Saves `.txt` files into:
```text
datasets\MM_Maker_Levels
```

## Create a filtered dataset (with quality filters and source tracking)

`--stride_x` and `--stride_y` control how far the scan window moves between samples; set both to the screen size (16/14) for non-overlapping, screen-aligned extraction. `--scan_mode snap` extracts scenes that snap to fully null-free screens. `--max_enemies N` drops scenes with more than `N` enemy tiles. `--min_content_pct P` drops scenes where less than `P`% of tiles are real content. `--include_moving_ground` includes moving-ground/platform tiles, excluded by default since their motion isn't represented in the static tileset graphics.

```bash
cd ..
python create_megaman_json_data.py --levels datasets\MM_Maker_Levels --tileset datasets\MM.json --stride_x 16 --stride_y 14 --scan_mode snap --max_enemies 4 --min_content_pct 15 --output datasets\MM_Levels_Filtered.json
```

Then generate deterministic captions for it:
```
python MM_create_ascii_captions.py --dataset datasets\MM_Levels_Filtered.json --tileset datasets\MM.json --output datasets\MM_LevelsAndCaptions-filtered-regular.json
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

Use `--game MM-Simple` instead if the dataset was generated with `--group_encodings`. For a quick test run instead of waiting on full training, add `--num_epochs 2 --save_image_epochs 100000`.

**Known limitation:** Moving-ground platforms (the `M` tile) are excluded by default rather than properly represented, since we don't yet have graphics or a static-scene encoding for their motion. See the GitHub issue tracking proper tileset/graphics support for moving-ground platforms for the planned fix.

## Generate Levels

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

## Mega Man Maker Conversion

Generated `.txt` files can be converted back into playable Mega Man Maker levels.

```bash
cd MM_Batch
MegaManMaker.bat
```

Drag and drop:
- `.mmlv` → converts to `.txt`
- `.txt` → converts to `.mmlv`

They appear in Mega Man Maker under `My Levels`.

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



## Train local text encoder

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

## Train unconditional model
Train an unconditional diffusion model without any text conditioning:
```
python train_diffusion.py --json datasets\MM_LevelsAndCaptions-simple-regular.json --augment --output_dir MM_unconditional_simple0 --seed 0 --game MM-Simple
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



## LLM captions (experimental)

Instead of the deterministic captions from `MM_create_ascii_captions.py`, you can choose to caption levels with an LLM. This produces five diverse natural-language captions per scene, grounded on pre-computed structural metadata about the scene. `--llm` picks the inference source: `ollama` runs a local model (default), while `claude` and `openai` call their respective APIs and require an API key in `.env`. `--levels` accepts either a JSON dataset from `create_megaman_json_data.py` or a directory of VGLC-ASCII `.txt` files. `--model` overrides the per-source default model (qwen3.5:9b for Ollama, Sonnet 4.6 for Claude, GPT-5.1 for OpenAI), and `--limit` caps how many scenes are captioned.

Example usage that captions the complete levels in the VGLC:
```bash
python llm_ascii_to_caption.py --levels ..\TheVGLC\MegaMan\Enhanced --tileset datasets\MM.json --llm ollama --output datasets\MM_LevelsAndLLMCaptions-full.json
```

### Training Models with LLM Captions
 
 To create a LLM-captioned dataset and train a conditional diffusion model with it, you can call `train-conditional-llm.bat`. It builds the level dataset with `create_megaman_json_data.py`, captions it with `llm_ascii_to_caption.py`, splits the result into train/validate/test sets, then trains a text-conditional diffusion model on the LLM captions using a general-purpose pretrained text encoder (from Hugging Face). 
```bash
cd MM_Batch
train-conditional-llm.bat 0 MiniLM [split]
```

It takes an optional seed (defaults to `0`), a pretrained text encoder (`MiniLM`, `GTE`, `CLIP`, or `T5`, defaults to `MiniLM`), and an optional `split` flag that gives each caption sentence its own embedding vector.

## Timing the pipeline

The training-pipeline batch files (`MM_conditional*.bat`, `MM_unconditional*.bat`, `train-conditional-llm.bat`) record how long each stage takes. After every major step (dataset creation, captioning, MLM training, etc.) the batch file appends a timestamp to a per-run log via `log_timestamp.py`. When the run finishes, the log is moved into the trained model's directory as `pipeline_timing.jsonl` (next to `training_log_.jsonl`).

A log entry looks like the following:
```json
{"timestamp": "2026-06-29 14:03:12", "event": "diffusion training", "status": "complete", "elapsed_seconds_since_prev": 8421.0, "prev_event": "MLM training"}
```

You can also log steps from your own custom training pipelines like so:
```bash
python log_timestamp.py --log_file timing_logs\my-run.jsonl --status start --event "pipeline start"
python log_timestamp.py --log_file timing_logs\my-run.jsonl --event "MMLV download"
```



## Mega Man Maker

[Mega Man Maker](https://github.com/schrum2/MarioDiffusion/tree/dev_alaaAlmzayen/megaman)