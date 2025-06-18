# Mario Diffusion

Generate Mario level scenes with a diffusion model conditioned on text input.

## Set up the repository

This repository can be checked out with this command:
```
git clone https://github.com/schrum2/MarioDiffusion.git
```
You will also need to check out level data from [My forked copy of TheVGLC](https://github.com/schrum2/TheVGLC) to create the training dataset:
```
git clone https://github.com/schrum2/TheVGLC.git
```
Both of these directories should be in the same parent directory. Next, enter the `MarioDiffusion` repository.
```
cd MarioDiffusion
```
Before running any code, install all requirements with pip:
```
pip install -r requirements.txt
```

## Create datasets

This batch file call will create sets of 16x16 level scenes of both SMB1 and SMB2 (Japan), as well as a combination of both. Afterwards, it will create captions for all 3 datasets, tokenizers for the data, test captions for later training, and finally splits the data into training, validation, and testing json files
```
batch\Mar1and2-data.bat
```
Now you can browse level scenes and their captions with a command like this (the json file can be replaced by any levels and captions json file in datasets):
```
python ascii_data_browser.py datasets\\Mar1and2_LevelsAndCaptions-regular.json 
```

## Train text encoder

Masked language modeling will be used to pre-train the text embedding model. Use whatever dataset you like with an appropriate tokenizer. It is reccomended to supply the validation and test datasets of the same type as well, though it is optional, and only used for evaluation.
```
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --test_json datasets\Mar1and2_LevelsAndCaptions-regular-test.json --pkl datasets\Mar1and2_Tokenizer-regular.pkl --output_dir Mar1and2-MLM-regular0 --seed 0
```
A report evaluating the accuracy of the final model on the training data is provided after training, but you can repeat a similar evaluation with this command:
```
python evaluate_masked_token_prediction.py --model_path Mar1and2-MLM-regular0 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json
```
You can also see how the accuracy on the training set changes throughout training by evaluating all checkpoints with this command:
```
python evaluate_masked_token_prediction.py --model_path Mar1and2-MLM-regular0 --compare_checkpoints --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json
```
To see accuracy on the validation set over time instead, run this command:
```
python evaluate_masked_token_prediction.py --model_path Mar1and2-MLM-regular0 --compare_checkpoints --json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json
```

## Train text-conditional diffusion model

Now that the text embedding model is ready, train a diffusion model conditioned on text embeddings from the descriptive captions:
```
python train_diffusion.py --save_image_epochs 20 --augment --text_conditional --output_dir Mar1and2-conditional-regular0 --num_epochs 500 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --pkl datasets\Mar1and2_Tokenizer-regular.pkl --mlm_model_dir Mar1and2-MLM-regular0 --plot_validation_caption_score --seed 0 
```
If you care more about speed than seeing intermediate results, you can set --save_image_epochs to an arbitrarily large number, like this
```
python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir Mar1and2-conditional-regular0 --num_epochs 500 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --pkl datasets\Mar1and2_Tokenizer-regular.pkl --mlm_model_dir Mar1and2-MLM-regular0 --plot_validation_caption_score --seed 0 
```
You can also train with negative prompting by adding an additional flag like this
```
python train_diffusion.py --save_image_epochs 20 --augment --text_conditional --output_dir Mar1and2-conditional-negative0 --num_epochs 500 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --pkl datasets\Mar1and2_Tokenizer-regular.pkl --mlm_model_dir Mar1and2-MLM-regular0 --plot_validation_caption_score --seed 0 --negative_prompt_training
```
You can swap out the dataset and, tokenizer, and language model however you like, as long as everything is consistent.

## Generate levels from text-conditional diffusion model

To generate unconditional levels (not based on text embeddings), use this batch file:
```
batch\run_diffusion_multi.bat Mar1and2-conditional-regular0 regular Mar1and2 text
```
This batch file automatically creates 2 different sets of 100 samples, one set that is 16 blocks wide, and another that is 128 blocks wide. If you'd like to run just one of these commands, or customize the output further, you can with this command:
```
python run_diffusion.py --model_path Mar1and2-conditional-regular0 --num_samples 100 --text_conditional --save_as_json --output_dir Mar1and2-conditional-regular0-unconditional-samples --level_width 16
```
Captions will be automatically assigned to the levels, and you can browse that data with this command:
```
python ascii_data_browser.py Mar1and2-conditional-regular0-unconditional-samples\all_levels.json
```
But to actually provide captions to guide the level generation, use this command
```
python text_to_level_diffusion.py --model_path Mar1and2-conditional-regular0
```
An easier-to-use GUI interface will let you select and combine known caption phrases to send to the model. Note that the selection of known phrases needs to come from the dataset you trained on.
```
python interactive_tile_level_generator.py datasets\Mar1and2_LevelsAndCaptions-regular-train.json Mar1and2-conditional-regular0
```
Interactively evolve level scenes in the latent space of the conditional model:
```
python evolve_interactive_conditional_diffusion.py --model_path Mar1and2-conditional-regular0
```
Automatically evolve level scenes in the latent space of the model (must put a caption into the quotations ex "full floor. one enemy."):
```
python evolve_automatic.py --model_path Mar1and2-conditional-regular0 --target_caption " "
```

## Evaluate caption adherence of text-conditional diffusion model

You can evaluate the final model's ability to adhere to input captions with this command:
```
python evaluate_caption_adherence.py --model_path Mar1and2-conditional-regular0 --save_as_json --json datasets\Mar1and2_LevelsAndCaptions-regular.json --output_dir text-to-level-final
```
You can also evaluate the how caption adherence changed during training with respect to the testing set:
```
python evaluate_caption_adherence.py --model_path Mar1and2-conditional-regular0 --save_as_json --json datasets\Mar1and2_LevelsAndCaptions-regular-test.json --compare_checkpoints 
```
However, it is easy to match the captions used during training. You can evaluate the how caption adherence changed during training with respect to a previously unseen randomly generated captions too:
```
python evaluate_caption_adherence.py --model_path Mar1and2-conditional-regular0 --save_as_json --json datasets\Mar1and2_RandomTest-regular.json --compare_checkpoints 
```
If you'd like to do all 3 of these commands at once (as well as automatically generate example level samples), you can do so by running the batch file like this:
```
batch\evaluate_caption_adherence_multi.bat Mar1and2-conditional-regular0 regular Mar1and2
```

## Train unconditional diffusion model

To train an unconditional diffusion model without any text embeddings, run this command:
```
python train_diffusion.py --augment --output_dir Mar1and2-unconditional0 --num_epochs 500 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --seed 0 
```
You can also use this batch file (it also 100 short and 100 long samples from the model once it's trained):
```
cd batch
train-unconditional.bat 0 Mar1and2 
```

## Generate levels from unconditional model

Just like with the text conditional model, you can get level samples from the batch file or a seperate command. The batch file still gets 2 sets of 100 samples, but the arguments are a little different
```
batch\run_diffusion_multi.bat Mar1and2-unconditional0 regular Mar1and2
```
As with before, to get more control, you can simply run this once from the command line
```
python run_diffusion.py --model_path Mar1and2-unconditional0 --num_samples 100 --save_as_json --output_dir Mar1and2-unconditional0-unconditional-samples --level_width 16
```
View the saved levels in the data browser
```
python ascii_data_browser.py Mar1and2-unconditional0-unconditional-samples\all_levels.json
```
Interactively evolve level scenes in the latent space of the unconditional model:
```
python evolve_interactive_unconditional_diffusion.py --model_path Mar1and2-unconditional0
```

## Train Generative Adversarial Network (GAN) model

GANs are an older technology, but they can also be trained to generate levels:
```
python train_wgan.py --augment --json datasets\Mar1and2_LevelsAndCaptions-regular.json --num_epochs 5000 --nz 32 --output_dir Mar1and2-wgan0 --seed 0
```
Just like with the diffusion model, you can save a little bit of time by cutting out intermediate results like this
```
python train_wgan.py --augment --json datasets\Mar1and2_LevelsAndCaptions-regular.json --num_epochs 5000 --nz 32 --output_dir Mar1and2-wgan0 --seed 0 --save_image_epochs 10000
```
You can also use the batch file instead (this will also generate levels with the wgan):
```
cd batch
train-wgan.bat 0 Mar1and2
```

## Generate levels from GAN

Create samples from the final GAN with this command (assuming the batch file hasn't already)
```
python run_wgan.py --model_path Mar1and2-wgan0\final_models\generator.pth" --num_samples 100 --output_dir Mar1and2-wgan0-samples --save_as_json
```
View the saved levels in the data browser
```
python ascii_data_browser.py wgan_samples\all_levels.json
```
Interactively evolve level scenes in the latent space of the GAN model:
```
python evolve_interactive_wgan.py --model_path Mar1and2-wgan0\final_models\generator.pth
```

## Train Five Dollar Model (FDM)
The five-dollar-model is a lightweight feedforward network that trains fast, but has a pretty small maximum performance. They can be trained with a call to the batch file, which will run metrics for you
```
cd batch
train-fdm.bat 0 Mar1and2 regular MiniLM
```
Alternatively, it can be trained individually like so
```
python train_fdm.py --augment --output_dir Mar1and2-fdm-MiniLM-regular0 --num_epochs 100 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --pretrained_language_model sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --plot_validation_caption_score --embedding_dim 384 --seed 0
```

## Generate levels from FDM

Create samples from an FDM with this command
```
python text_to_level_fdm.py --model_path Mar1and2-fdm-MiniLM-regular0
```

## Comparing model results

TODO

## Generating MarioGPT data for comparison

Most of the MarioGPT data is taken care of in this batch file, which can be run like this
```
cd batch
MarioGPT-data.bat
```
This batch file generates 96 levels of size 128 using MarioGPT, stores, pads and captions them in the same format as our unconditional models, and then runs metrics on both sliced 16x16 level samples, as well as the full 16x128 generated levels.  

If you'd like to do each of these steps seperatly, that can be done with this series of commands:

First, the level generation can be done with this command, which saves generated levels in a new folder called MarioGPT_Levels, in both text and image format.
```
python run_gpt2.py --output_dir "MarioGPT_Levels" --num_collumns 128
```
Afterwards, this command will take those levels, pad them, and store them in new files in the datasets directory. (The stride variable controls how long individual segments are, the batch file runs this twice to get levels of length 128 and 16)
```
python create_level_json_data.py --output "datasets\\MarioGPT_Levels.json" --levels "MarioGPT_Levels\levels" --stride 16
```
Afterwards, we use this command to give captions to these levels
```
python create_ascii_captions.py --dataset "datasets\\MarioGPT_Levels.json" --output "datasets\\MarioGPT_LevelsAndCaptions-regular.json"
```
And then, lastly, we can use this command to get metrics on the generated levels
```
python calculate_gpt2_metrics.py --generated_levels "datasets\\MarioGPT_LevelsAndCaptions-regular.json" --training_levels "datasets\\Mar1and2_LevelsAndCaptions-regular.json" --output_dir "MarioGPT_metrics"
```