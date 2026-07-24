# Lode Runner Diffusion

Generate Lode Runner levels with a diffusion model conditioned on text input.
The Lode Runner models produced by this code are not as impressive as the Mario Diffusion models.
Our main therory as to why is that Lode Runner has a much smaller dataset with only 150 samples.
The samples also contain more diversity than the Mario data,
and the game itself involves more complex functional
requirements in order for levels to be beatable.
Still, you can train diffusion models that generate
Lode Runner levels, and play the generated levels.
These instructions assume you have followed the basic setup instructions in the main [README](../README.md) first.

## Batch Files

Our code was developed on Windows machines, so we have made extensive use of batch files for convenience. However, these will not work on Linux/Mac systems. The Python scripts that are called from these batch files should work on any system, though this has not been fully tested. The instructions below describe how to use the batch files and certain Python scripts, but you can look in the batch files to execute individual commands as needed.

## Create datasets

Data for training Lode Runner models 
is not in the repo, but it can easily be constructed using
data from [my forked copy of TheVGLC](https://github.com/schrum2/TheVGLC).
Note that the following
command should be executed in the parent directory of the `MarioDiffusion` repository so that
the directories for `MarioDiffusion` and `TheVGLC` are next to each other in the same directory:
```
git clone https://github.com/schrum2/TheVGLC.git
```
Once you have my version of `TheVGLC` and `MarioDiffusion`, go into the `Game_LR/BATCH` sub-directory in the
`MarioDiffusion` repo for Lode Runner batch files.
```
cd MarioDiffusion
cd Game_LR
cd BATCH
```
Next, run a batch file to create datasets from the VGLC data. This batch file call will create
a json data set of 32x32 levels from the VGLC data for Lode Runner. Note that the  
top 10 rows are filled with blank space to extend
the height to 32, which is suitable for the diffusion
models we train.
Afterwards, it will create captions for the dataset, tokenizers for the data, random test captions for later evaluation, and finally splits the data into training, validation, and testing json files. 
Run this command:
```
LR-data.bat
```
Now you can browse level scenes and their captions with a command like this (the first json file can be replaced by any levels and captions json file in datasets):
```
python ascii_data_browser.py Game_LR/DATA/LR_LevelsAndCaptions-regular.json LR
```
This is not required, but will give you insight into the data.

## Complete training and evaluation sequence

To train a text conditional diffusion model for Lode Runner, you should go to the batch directory first:
```
cd batch
```
Once here, you can train both a text encoder and its corresponding diffusion model back to back with a single command like this:
```
train-conditional.bat 0 LR regular LR
```
This is the exact same batch file used to train models for Mario, and there are only a few minor differences in the process when training a model for Lode Runner. For more details, see the batch file's contents.
You'll see that after training, extra evaluation of the produced model is carried out.

The core training steps that occur in the batch file are the training of the text encoder and the diffusion model.
Masked language modeling is used to train the text embedding model. 
The following command line will train a text embedding model based on the Lode Runner data created before:
```
python train_mlm.py --epochs 80000 --save_checkpoints --json Game_LR\DATA\LR_LevelsAndCaptions-regular-train.json --val_json Game_LR\DATA\LR_LevelsAndCaptions-regular-validate.json --test_json Game_LR\DATA\LR_LevelsAndCaptions-regular-test.json --pkl Game_LR\DATA\LR_Tokenizer-regular.pkl --output_dir LR-LR-MLM-regular0 --seed 0
```
After training the text embedding model, you can train a diffusion model conditioned on text embeddings from the descriptive captions:
```
python train_diffusion.py --augment --text_conditional --output_dir "LR-LR-conditional-regular0" --num_epochs 3000 --json Game_LR\DATA\LR_LevelsAndCaptions-regular-train.json --val_json Game_LR\DATA\LR_LevelsAndCaptions-regular-validate.json --pkl Game_LR\DATA\LR_Tokenizer-regular.pkl --mlm_model_dir LR-LR-MLM-regular0 --plot_validation_caption_score --seed 0 --game LR
```
You can also train a Lode Runner model using a pre-trained text encoder instead of training your own MLM transformer.
Here is the easy way to launch the training and evaluation with a batch file:
```
train-conditional-pre.bat 0 LR regular LR MiniLM split
```
This command trains one diffusion model that uses `MiniLM` as its text model, and the `split` parameter means that individual phrases from the Lode Runner captions each get their own embedding vector. You can simply leave the `split` out to embed each caption with a single vector, and you can also swap `MiniLM` with `GTE` or other models mentioned in the batch file.
You can also use the `train_diffusion.py` script directly to train a model however you like.

## Generate levels from text-conditional diffusion model

These options are similar to what you can do with Mario levels.
To generate unconditional levels (not based on text embeddings), use this batch file:
```
batch\run_diffusion_multi.bat LR-LR-conditional-regular0 regular LR
```
When used with Lode Runner, this batch file makes one set of 100 samples. It is essentially just running this command:
```
python run_diffusion.py --model_path LR-LR-conditional-regular0 --num_samples 100 --save_as_json --output_dir LR-LR-conditional-regular0-unconditional-samples --game LR
```
Captions will be automatically assigned to the levels, and you can browse that data with this command:
```
python ascii_data_browser.py LR-LR-conditional-regular0-unconditional-samples\all_levels.json LR
```
But to actually provide captions to guide the level generation, use this command
```
python text_to_level_diffusion.py --model_path LR-LR-conditional-regular0 --game LR
```
Similarly, the GUI used with Mario can also be used with Lode Runner, like so:
```
python interactive_tile_level_generator.py --model_path LR-LR-conditional-regular0 --load_data Game_LR/DATA/LR_LevelsAndCaptions-regular.json --game LR
```
As with Mario, additional settings are recommended when working with models trained on absence captions.

You can also interactively evolve level scenes in the latent space of the conditional model:
```
python evolve_interactive_conditional_diffusion.py --model_path LR-LR-conditional-regular0 --game LR
```

## Train unconditional diffusion models

To train an unconditional diffusion model without any text embeddings, run this batch file:
```
cd batch
train-unconditional.bat 0 LR LR
```

## Generate levels from unconditional model

Just like with the text conditional model, you can get level samples from the batch file or a seperate command.
```
batch\run_diffusion_multi.bat LR-LR-unconditional0 regular LR
```
As before, to get more control, you can simply run this from the command line
```
python run_diffusion.py --model_path LR-LR-unconditional0 --num_samples 100 --save_as_json --output_dir LR-LR-unconditional0-unconditional-samples --game LR
```
View the saved levels in the data browser
```
python ascii_data_browser.py LR-LR-unconditional0-unconditional-samples\all_levels.json LR
```
Interactively evolve level scenes in the latent space of the unconditional model:
```
python evolve_interactive_unconditional_diffusion.py --model_path LR-LR-unconditional0 --game LR
```

## Train Generative Adversarial Network (GAN) model

GANs can also be trained for Lode Runner. Just use this batch file:
```
cd batch
train-wgan.bat 0 LR LR
```

## Generate levels from GAN

Create samples from the final GAN with this command (assuming the batch file hasn't already)
```
python run_wgan.py --model_path LR-LR-wgan0\final_models\generator.pth --num_samples 100 --output_dir LR-LR-wgan0-samples --save_as_json --game LR
```
View the saved levels in the data browser
```
python ascii_data_browser.py LR-LR-wgan0-samples\all_levels.json LR
```
Interactively evolve level scenes in the latent space of the GAN model:
```
python evolve_interactive_wgan.py --model_path LR-LR-wgan0\final_models\generator.pth --game LR
```

## Citation

The results with Lode Runner are admittedly less impressive than our results in Mario, which is part of the reason they have not yet appeared in any of our publications. The Lode Runner dataset is smaller, more varied, and harder to adequately describe with deterministically assigned captions. Still, if our code is in some way useful to you, then you could still cite this repo:

```bibtex
@misc{schrum:loderunnerdiffusion,
  author       = {Schrum, Jacob and Williams, Reid},
  title        = {Lode Runner Diffusion},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/schrum2/MarioDiffusion}},
  note         = {Lode Runner code in the MarioDiffusion repository}
}
```