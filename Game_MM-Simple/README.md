# Mega Man Diffusion

Generate Mega Man level scenes with a diffusion model conditioned on text input.
These instructions focus on using VGLC data to train a diffusion model to create Mega Man levels. The tileset used is simplified in a manner similar to the Mario Diffusion results, where all enemies map to a single common tile. Therefore, the complexity of the tileset is comparable to Mario, but Mega Man levels are a bit more complex because they progress not only from left to right, but sometimes vertically as well.
This code will allow you to train a diffusion model that generates Mega Man scenes, and combine them into levels that are playable in [Mega Man Maker](https://megamanmaker.com/).
These instructions assume you have followed the basic setup instructions in the main [README](../README.md) first.

## Batch Files

Our code was developed on Windows machines, so we have made extensive use of batch files for convenience. However, these will not work on Linux/Mac systems. The Python scripts that are called from these batch files should work on any system, though this has not been fully tested. The instructions below describe how to use the batch files and certain Python scripts, but you can look in the batch files to execute individual commands as needed.

## Create datasets

Data for training Mega Man models 
is not in the repo, but it can easily be constructed using
data from [my forked copy of TheVGLC](https://github.com/schrum2/TheVGLC).
Note that the following
command should be executed in the parent directory of the `MarioDiffusion` repository so that
the directories for `MarioDiffusion` and `TheVGLC` are next to each other in the same directory:
```
git clone https://github.com/schrum2/TheVGLC.git
```
Once you have my version of `TheVGLC` and `MarioDiffusion`, go into the `Game_MM-Simple/BATCH` sub-directory in the
`MarioDiffusion` repo for Lode Runner batch files.
```
cd MarioDiffusion
cd Game_MM-Simple
cd BATCH
```
Next, run a batch file to create datasets from the VGLC data. This batch file call will create
a json data set of level scenes from Mega Man 1 levels in the VGLC data. Note that the  
top 2 rows of each scene are filled with blank space to extend
the height from 14 to 16, which is suitable for the diffusion
models we train.
Afterwards, the batch file will create captions for the dataset, a tokenizer for the data, random test captions for later evaluation, and finally splits the data into training, validation, and testing json files. 
Run this command:
```
MM-Simple-data.bat
```
Now you can browse level scenes and their captions with a command like this (the first json file can be replaced by any levels and captions json file in datasets):
```
python ascii_data_browser.py Game_MM-Simple/DATA/MM-Simple_LevelsAndCaptions-regular.json MM-Simple
```
This is not required, but will give you insight into the data.

## Complete training and evaluation sequence

To train a text conditional diffusion model for the simplified Mega Man tileset, you should go to the batch directory first:
```
cd batch
```
Once here, you can train both a text encoder and its corresponding diffusion model back to back with a single command like this:
```
train-conditional.bat 0 MM-Simple regular MM-Simple
```
This is the exact same batch file used to train models for Mario.
You'll see that after training, extra evaluation of the produced model is carried out.

The core training steps that occur in the batch file are the training of the text encoder and the diffusion model.
Masked language modeling is used to train the text embedding model. 
The following command line will train a text embedding model based on the Lode Runner data created before:
```
python train_mlm.py --epochs 300 --save_checkpoints --json Game_MM-Simple/DATA/MM-Simple_LevelsAndCaptions-regular-train.json --val_json Game_MM-Simple/DATA/MM-Simple_LevelsAndCaptions-regular-validate.json --test_json Game_MM-Simple/DATA/MM-Simple_LevelsAndCaptions-regular-test.json --pkl Game_MM-Simple/DATA/MM-Simple_Tokenizer-regular.pkl --output_dir MM-Simple-MM-Simple-MLM-regular0 --seed 0
```
After training the text embedding model, you can train a diffusion model conditioned on text embeddings from the descriptive captions:
```
python train_diffusion.py --save_image_epochs 20 --text_conditional --output_dir MM-Simple-MM-Simple-conditional-regular0 --num_epochs 500 --json Game_MM-Simple/DATA/MM-Simple_LevelsAndCaptions-regular-train.json --val_json Game_MM-Simple/DATA/MM-Simple_LevelsAndCaptions-regular-validate.json --pkl Game_MM-Simple/DATA/MM-Simple_Tokenizer-regular.pkl --mlm_model_dir MM-Simple-MM-Simple-MLM-regular0 --plot_validation_caption_score --seed 0
```
You can also train a Mega Man model using a pre-trained text encoder instead of training your own MLM transformer.
Here is the easy way to launch the training and evaluation with a batch file:
```
train-conditional-pre.bat 0 MM-Simple regular MM-Simple MiniLM split
```
This command trains one diffusion model that uses `MiniLM` as its text model, and the `split` parameter means that individual phrases from the Mega Man captions each get their own embedding vector. You can simply leave the `split` out to embed each caption with a single vector, and you can also swap `MiniLM` with `GTE` or other models mentioned in the batch file.
You can also use the `train_diffusion.py` script directly to train a model however you like.

## Generate levels from text-conditional diffusion model

These options are similar to what you can do with Mario levels.
To generate unconditional levels (not based on text embeddings), use this batch file:
```
batch\run_diffusion_multi.bat MM-Simple-MM-Simple-conditional-regular0 regular MM-Simple
```
This creates both small and long level samples. Creating small unconditional level scenes can be done with this command:
```
python run_diffusion.py --model_path MM-Simple-MM-Simple-conditional-regular0 --num_samples 100 --save_as_json --output_dir MM-Simple-MM-Simple-conditional-regular0-unconditional-samples --game MM-Simple
```
Captions will be automatically assigned to the levels, and you can browse that data with this command:
```
python ascii_data_browser.py MM-Simple-MM-Simple-conditional-regular0-unconditional-samples\all_levels.json MM-Simple
```
But to actually provide captions to guide the level generation, use this command
```
python text_to_level_diffusion.py --model_path MM-Simple-MM-Simple-conditional-regular0 --game MM-Simple
```
Similarly, the GUI used with Mario can also be used with Mega Man, like so:
```
python interactive_tile_level_generator.py --model_path MM-Simple-MM-Simple-conditional-regular0 --load_data Game_MM-Simple/DATA/MM-Simple_LevelsAndCaptions-regular.json --game MM-Simple
```
However, there are some new options here that are specific to Mega Man. If you add several generated level scenes
to a constructed level, you can then click the button to Build a Mega Man level, which will bring up a 2D grid
layout where you can layout the scenes into a complete level. 

![ArrangeMegaManMap.png](Building Mega Man Level)

You can even choose to play this constructed level
in [Mega Man Maker](https://megamanmaker.com/) if you have it installed.

You can also interactively evolve level scenes in the latent space of the conditional model:
```
python evolve_interactive_conditional_diffusion.py --model_path MM-Simple-MM-Simple-conditional-regular0 --game MM-Simple
```

## Train unconditional diffusion models

To train an unconditional diffusion model without any text embeddings, run this batch file:
```
cd batch
train-unconditional.bat 0 MM-Simple MM-Simple
```

## Generate levels from unconditional model

Just like with the text conditional model, you can get level samples from the batch file or a seperate command.
```
batch\run_diffusion_multi.bat MM-Simple-MM-Simple-unconditional0 regular LR
```
As before, to get more control, you can simply run this from the command line
```
python run_diffusion.py --model_path MM-Simple-MM-Simple-unconditional0 --num_samples 100 --save_as_json --output_dir MM-Simple-MM-Simple-unconditional0-unconditional-samples --game MM-Simple
```
View the saved levels in the data browser
```
python ascii_data_browser.py MM-Simple-MM-Simple-unconditional0-unconditional-samples\all_levels.json MM-Simple
```
Interactively evolve level scenes in the latent space of the unconditional model:
```
python evolve_interactive_unconditional_diffusion.py --model_path MM-Simple-MM-Simple-unconditional0 --game MM-Simple
```
Note that the Mega Man Level editor can also be invoked from the interactive evolution interface.

## Train Generative Adversarial Network (GAN) model

GANs can also be trained for Mega Man. Just use this batch file:
```
cd batch
train-wgan.bat 0 MM-Simple MM-Simple
```

## Generate levels from GAN

Create samples from the final GAN with this command (assuming the batch file hasn't already)
```
python run_wgan.py --model_path MM-Simple-MM-Simple-wgan0\final_models\generator.pth --num_samples 100 --output_dir MM-Simple-MM-Simple-wgan0-samples --save_as_json --game MM-Simple
```
View the saved levels in the data browser
```
python ascii_data_browser.py MM-Simple-MM-Simple-wgan0-samples\all_levels.json MM-Simple
```
Interactively evolve level scenes in the latent space of the GAN model:
```
python evolve_interactive_wgan.py --model_path MM-Simple-MM-Simple-wgan0\final_models\generator.pth --game MM-Simple
```











CHANGE BELOW THIS








Train an unconditional diffusion model without any text embeddings:

This entire process — from creating the level sample files, through captioning,
tokenizing, splitting, and training — can be done with this batch file:

```
cd MM_Batch
MM_unconditional.bat [size]
```
`size` is optional and sets both the scene width and height (default 16 wide,
14 tall if omitted; passing e.g. `32` makes both dimensions 32). Output model
directory is named `MM-simple{size}-unconditional0`.

```
python train_diffusion.py --json datasets\MM_LevelsAndCaptions-simple-regular.json --augment --output_dir MM_unconditional_simple0 --seed 0 --game MM-Simple
```





















## Train and generate levels with block2vec tile embeddings (experimental)

By default, unconditional diffusion models represent each tile as a one-hot vector. Block2Vec replaces this with learned embedding vectors trained on 3x3 tile windows, so contextually similar tiles end up with similar vectors.

```
MM_Batch\MM_unconditional-embedding.bat {embedding_dims}
```

(`embedding_dims` is optional, default 16.)

Manual steps:

Slice the VGLC levels into 3x3 tile windows for embedding training:
```
python create_tile_level_json_data.py --tileset Game_MM-Simple/MM-Simple-tileset.json --levels ..\TheVGLC\MegaMan\Enhanced --output datasets\MM_3x3_Tiles-simple.json --tile_size 3 --char_map datasets\MM-VGLC-to-simple.json
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




















## Embedding

Slice the VGLC levels into 3x3 tile windows for embedding training:
```
python create_tile_level_json_data.py --tileset Game_MM-Simple/MM-Simple-tileset.json --levels ..\TheVGLC\MegaMan\Enhanced --output datasets\MM_3x3_Tiles-simple.json --tile_size 3 --char_map datasets\MM-VGLC-to-simple.json
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










## Mega Man Maker

[Mega Man Maker](https://github.com/schrum2/MarioDiffusion/tree/dev_alaaAlmzayen/megaman)



