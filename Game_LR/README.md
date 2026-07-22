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

You can also train a Lode Runner model using a pre-trained text encoder instead of training your own MLM transformer.
Here is an example:
```
train-conditional-pre.bat 0 LR regular LR MiniLM split
```
This command trains one diffusion model that uses `MiniLM` as its text model, and the `split` parameter means that individual phrases from the Lode Runner captions each get their own embedding vector. You can simply leave the `split` out to embed each caption with a single vector, and you can also swap `MiniLM` with `GTE` or other models mentioned in the batch file.

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
python ascii_data_browser.py LR-LR-unconditional0-unconditional-samples\all_levels.json
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
python ascii_data_browser.py LR-LR-wgan0-samples\all_levels.json
```
Interactively evolve level scenes in the latent space of the GAN model:
```
python evolve_interactive_wgan.py --model_path LR-LR-wgan0\final_models\generator.pth --game LR
```

## Conclusion










Actually, incorporate some of the instructions below into those above







## Generating and playing Lode Runner levels
If the user wants to see the captions and play all of the original levels, use the following command line.
All of the levels should be playable and beatable with how Lode Runner is currently played. 
If the user wishes to quit playing a level, they can use the 'q' key which should close the current game window
allowing them to reuse the data browser again:
```
python ascii_data_browser.py datasets\LR_LevelsAndCaptions-regular.json Game_LR/LodeRunner.json
```

If the user wanted to play the levels without seeing the captions or level makeup, use the following command line. 
The following line allows the user to play the first level. If the user wants to play a different level, change the 1 to the level they wish to play. Must be in the MarioDiffusion directory to play:
```
python -m loderunner.main datasets\LR_LevelsAndCaptions-regular.json 1
```

But to actually provide captions to guide the level generation, use this command:
```
python text_to_level_diffusion.py --model_path LR-conditional-regular0 --game LR
```

An easier-to-use GUI interface will let you select and combine known caption phrases to send to the model. Note that the selection of known phrases needs to come from the dataset you trained on:
```
python interactive_tile_level_generator.py --load_data datasets\LR_LevelsAndCaptions-regular.json --model_path LR-conditional-regular0 --game LR 
```

## Train text encoder

Masked language modeling is used to train the text embedding model. Use whatever dataset you like with an appropriate tokenizer. It is recommended to supply the validation and test datasets of the same type as well, though it is optional, and only used for evaluation.

The following command line will train a text embedding model based on the Lode Runner data created before:
```
python train_mlm.py --epochs 80000 --save_checkpoints --json datasets\LR_LevelsAndCaptions-regular-train.json --val_json datasets\LR_LevelsAndCaptions-regular-validate.json --test_json datasets\LR_LevelsAndCaptions-regular-test.json --pkl datasets\LR_Tokenizer-regular.pkl --output_dir LR-MLM-regular0 --seed 0
```
A report evaluating the accuracy of the final model on the training data is provided after training, but you can repeat a similar evaluation with this command:
```
python evaluate_masked_token_prediction.py --model_path LR-MLM-regular0 --json datasets\LR_LevelsAndCaptions-regular-train.json
```
You can also see how the accuracy on the training set changes throughout training by evaluating all checkpoints with this command:
```
python evaluate_masked_token_prediction.py --model_path LR-MLM-regular0 --json datasets\LR_LevelsAndCaptions-regular-train.json --compare_checkpoints
```
To see accuracy on the validation set over time instead, run this command:
```
python evaluate_masked_token_prediction.py --model_path LR-MLM-regular0 --compare_checkpoints --json datasets\LR_LevelsAndCaptions-regular-validate.json
```

## Train text-conditional diffusion model

Now that the text embedding model is ready, train a diffusion model conditioned on text embeddings from the descriptive captions. Note that this can take a while:
```
python train_diffusion.py --augment --text_conditional --output_dir "LR-conditional-regular0" --num_epochs 3000 --json datasets\LR_LevelsAndCaptions-regular-train.json --val_json datasets\LR_LevelsAndCaptions-regular-validate.json --pkl datasets\LR_Tokenizer-regular.pkl --mlm_model_dir LR-MLM-regular0 --plot_validation_caption_score --seed 0 --game LR
```
Another trick if you care more about speed than seeing intermediate results is to set `--save_image_epochs` to a large number (larger than the number of epochs), like this
```
python train_diffusion.py --save_image_epochs 10000 --augment --text_conditional --output_dir "LR-conditional-regular0" --num_epochs 3000 --json datasets\LR_LevelsAndCaptions-regular-train.json --val_json datasets\LR_LevelsAndCaptions-regular-validate.json --pkl datasets\LR_Tokenizer-regular.pkl --mlm_model_dir LR-MLM-regular0 --plot_validation_caption_score --seed 0 --game LR
```
You can also train with negative prompting by adding an additional flag like this
```
python train_diffusion.py --save_image_epochs 20 --augment --text_conditional --output_dir "LR-conditional-regular0" --num_epochs 3000 --json datasets\LR_LevelsAndCaptions-regular-train.json --val_json datasets\LR_LevelsAndCaptions-regular-validate.json --pkl datasets\LR_Tokenizer-regular.pkl --mlm_model_dir LR-MLM-regular0 --plot_validation_caption_score --seed 0 --game LR --negative_prompt_training
```
You can also use this batch file which will train the text embedding model, train a conditional diffusion model,
and generate unconditional levels not based on text embeddings:
```
cd LR-batch
LR-train-conditional.bat 0  
```

## Generate levels from text-conditional diffusion model

To generate unconditional levels (not based on text embeddings), use this command line:
```
python run_diffusion.py --model_path LR-conditional-regular0 --num_samples 100 --text_conditional --save_as_json --output_dir "LR-conditional-regular0-unconditional-samples" --game LR
```
Captions will be automatically assigned to the levels, and you can browse that data with this command:
```
python ascii_data_browser.py LR-conditional-regular0-unconditional-samples\all_levels.json
```
But to actually provide captions to guide the level generation, use this command
```
python text_to_level_diffusion.py --model_path LR-conditional-regular0 --game LR
```
An easier-to-use GUI interface will let you select and combine known caption phrases to send to the model. Note that the selection of known phrases needs to come from the dataset you trained on.
```
python interactive_tile_level_generator.py --model_path LR-conditional-regular0 --load_data datasets/LR_LevelsAndCaptions-regular.json --game LR
```
Interactively evolve level scenes in the latent space of the conditional model:
```
python evolve_interactive_conditional_diffusion.py --model_path LR-conditional-regular0 --game LR
```

## Train unconditional diffusion model

To train an unconditional diffusion model without any text embeddings, run this command:
```
python train_diffusion.py --augment --output_dir "LR-unconditional0" --num_epochs 3000 --json datasets\LR_LevelsAndCaptions-regular-train.json --val_json datasets\LR_LevelsAndCaptions-regular-validate.json --seed 0 --game LR
```
You can also use this batch file which will train an unconditional diffusion model and generate unconditional 
levels not based on text embeddings:
```
cd LR-batch
LR-train-unconditional.bat 0  
```

## Generate levels from unconditional model

To generate 100 unseen Lode Runner samples, you can simply run this once from the command line:
```
python run_diffusion.py --model_path LR-unconditional0 --num_samples 100 --save_as_json --output_dir LR-unconditional0-unconditional-samples
```
View the saved levels in the data browser:
```
python ascii_data_browser.py LR-unconditional0-unconditional-samples\all_levels.json
```
Interactively evolve level scenes in the latent space of the unconditional model:
```
python evolve_interactive_unconditional_diffusion.py --model_path LR-unconditional0 --game LR
```

## Train Generative Adversarial Network (GAN) model

GANs are an older technology, but they can also be trained to generate levels:
```
python train_wgan.py --augment --json datasets\LR_LevelsAndCaptions-regular.json --num_epochs 20000 --nz 10 --output_dir "LR-wgan0" --seed 0 --save_image_epochs 20 --game LR
```
Just like with the diffusion model, you can save a little bit of time by cutting out intermediate results like this
```
python train_wgan.py --augment --json datasets\LR_LevelsAndCaptions-regular.json --num_epochs 20000 --nz 10 --output_dir "LR-wgan0" --seed 0 --save_image_epochs 100000 --game LR

```
You can also use the batch file instead (this will also generate levels with the wgan):
```
cd LR-batch
train-wgan.bat 0 
```

## Generate levels from GAN

Create samples from the final GAN with this command (assuming the batch file hasn't already)
```
python run_wgan.py --model_path "LR-wgan0\final_models\generator.pth" --num_samples 100 --output_dir "LR-wgan0-samples" --save_as_json --game LR --nz 10
```
View the saved levels in the data browser
```
python ascii_data_browser.py LR-wgan_samples\all_levels.json
```
Interactively evolve level scenes in the latent space of the GAN model:
```
python evolve_interactive_wgan.py --model_path LR-wgan0\final_models\generator.pth --game LR --nz 10
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
