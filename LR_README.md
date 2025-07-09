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

## Train text encoder

Masked language modeling is used to train the text embedding model. Use whatever dataset you like with an appropriate tokenizer. It is reccomended to supply the validation and test datasets of the same type as well, though it is optional, and only used for evaluation.
```
python train_mlm.py --epochs 100000 --save_checkpoints --json datasets\LR_LevelsAndCaptions-regular-train.json --val_json datasets\LR_LevelsAndCaptions-regular-validate.json --test_json datasets\LR_LevelsAndCaptions-regular-test.json --pkl datasets\LR_Tokenizer-regular.pkl --output_dir LR-MLM-regular0 --seed 0
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

Now that the text embedding model is ready, train a diffusion model conditioned on text embeddings from the descriptive captions. Note that this can take a while. We used relatively modest consumer GPUs, so our models took about 12 hours to train:
```
python train_diffusion.py --augment --text_conditional --output_dir "LR-conditional-regular0" --num_epochs 25000 --json datasets\LR_LevelsAndCaptions-regular-train.json --val_json datasets\LR_LevelsAndCaptions-regular-validate.json --pkl datasets\LR_Tokenizer-regular.pkl --mlm_model_dir LR-MLM-regular0 --plot_validation_caption_score --seed 0 --game LR
```
Another trick if you care more about speed than seeing intermediate results is to set `--save_image_epochs` to a large number (larger than the number of epochs), like this
```
python train_diffusion.py --save_image_epochs 100000 --augment --text_conditional --output_dir "LR-conditional-regular0" --num_epochs 25000 --json datasets\LR_LevelsAndCaptions-regular-train.json --val_json datasets\LR_LevelsAndCaptions-regular-validate.json --pkl datasets\LR_Tokenizer-regular.pkl --mlm_model_dir LR-MLM-regular0 --plot_validation_caption_score --seed 0 --game LR
```
You can also train with negative prompting by adding an additional flag like this
```
python train_diffusion.py --save_image_epochs 20 --augment --text_conditional --output_dir "LR-conditional-regular0" --num_epochs 25000 --json datasets\LR_LevelsAndCaptions-regular-train.json --val_json datasets\LR_LevelsAndCaptions-regular-validate.json --pkl datasets\LR_Tokenizer-regular.pkl --mlm_model_dir LR-MLM-regular0 --plot_validation_caption_score --seed 0 --game LR --negative_prompt_training
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
python train_diffusion.py --augment --output_dir "LR-unconditional0" --num_epochs 25000 --json datasets\LR_LevelsAndCaptions-regular-train.json --val_json datasets\LR_LevelsAndCaptions-regular-validate.json --seed 0 --game LR
```
You can also use this batch file:
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
