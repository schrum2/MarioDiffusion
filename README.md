# Mario Diffusion

Generate Mario level scenes with a diffusion model conditioned on text input.

## Set up the repository

This repository can be checked out with this command:
```
git clone https://github.com/schrum2/MarioDiffusion.git
```
You will also need to check out level data from [TheVGLC](https://github.com/TheVGLC/TheVGLC) to create the training dataset:
```
git clone https://github.com/TheVGLC/TheVGLC.git
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

Extract a json data set of 16 by 16 level scenes from the VGLC data for Super Mario Bros with a command like this:
```
python create_level_json_data.py --output "SMB1_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros\\Processed"
```
You can also extract data from Super Mario Bros 2 (Japan) and Super Mario World:
```
python create_level_json_data.py --output "SMB2_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros 2 (Japan)\\Processed"
python create_level_json_data.py --output "SML_Levels.json"  --levels "..\\TheVGLC\\Super Mario Land\\Processed"
```
You can combine the data from the three Mario games into a single dataset:
```
python combine_data.py Mario_Levels.json SMB1_Levels.json SMB2_Levels.json SML_Levels.json
```
These files only contains the level scenes. Create captions for all level scenes with commands like this:
```
python create_ascii_captions.py --dataset SMB1_Levels.json --output SMB1_LevelsAndCaptions-regular.json
```
You can also make the captions explicitly mention things that are absent from each scene with the `--describe_absence` flag:
```
python create_ascii_captions.py --dataset SMB1_Levels.json --output SMB1_LevelsAndCaptions-absence.json --describe_absence
```
Now you can browse the level scenes and their captions with these commands:
```
python ascii_data_browser.py SMB1_LevelsAndCaptions-regular.json 
python ascii_data_browser.py SMB1_LevelsAndCaptions-absence.json 
```
It can also be useful to have a separate dataset of captions which are not use for training. The code used later supports splitting the data into separate sets for training, validation, and testing, but you can also make datasets of random captions with commands like this:
```
python create_validation_captions.py --save_file "SMB1_ValidationCaptions-regular.json" --json SMB1_LevelsAndCaptions-regular.json --seed 0
python create_validation_captions.py --save_file "SMB1_ValidationCaptions-absence.json" --json SMB1_LevelsAndCaptions-absence.json --seed 0 --describe_absence
```
You don't necessarily need to run all of these command individually. Simply running the batch file `BAT_datasets.bat` should create all the datasets you could need.

## Can I also get Mega Man Data? (TODO)

This doesn't work yet
```
python create_level_json_data.py --output "MM_Levels.json" --levels "..\\TheVGLC\\MegaMan"
```

## Train text encoder

First create a tokenizer for the caption data you want to train on. Most of these datasets have the same vocabulary, but there is a clear difference between datasets that describe the absence of entities and those that do not. Also, SMB1 has no upside down pipes, but these are present in the other games. The `BAT_datasets.bat` already creates a tokenizer for each dataset, but if you make a tokenizer for all of the Mario data, you should be covered:
```
python tokenizer.py save --json_file Mario_LevelsAndCaptions-regular.json    --pkl_file Mario_Tokenizer-regular.pkl
python tokenizer.py save --json_file Mario_LevelsAndCaptions-absence.json    --pkl_file Mario_Tokenizer-absence.pkl
```
Now, masked language modeling will be used to pre-train the text embedding model. Use whatever dataset you like with an appropriate tokenizer. The `--split` flag splits the data into training, validation, and testing, and also implements early stopping based on validation loss.
```
python train_mlm.py --epochs 300 --save_checkpoints --json SMB1_LevelsAndCaptions-regular.json --pkl SMB1_Tokenizer-regular.pkl --output_dir SMB1-MLM-regular --split
```
A report evaluating the accuracy of the final model on the training data is provided after training, but you can repeat a similar evaluation with this command:
```
python evaluate_masked_token_prediction.py --model_path SMB1-MLM-regular --json SMB1_LevelsAndCaptions-regular.json
```
You can also see how the accuracy on the training set changes throughout training by evaluating all checkpoints with this command:
```
python evaluate_masked_token_prediction.py --model_path SMB1-MLM-regular --compare_checkpoints --json SMB1_LevelsAndCaptions-regular.json
```
To see accuracy on the validation set over time instead, run this command:
```
python evaluate_masked_token_prediction.py --model_path SMB1-MLM-regular --compare_checkpoints --json SMB1_ValidationCaptions-regular.json
```

## Train text-to-level model

Now that the text embedding model is ready, train a diffusion model conditioned on text embeddings from the descriptive captions:
```
python train_diffusion.py --augment --text_conditional --output_dir "conditional-model" --num_epochs 200
```
You can train on just the Mario 2 data with this command:
```
python train_diffusion.py --augment --text_conditional --output_dir "SMB2-conditional-model" --num_epochs 200 --json SMB2_LevelsAndCaptions.json --mlm_model_dir mlm2 --pkl SMB2_Tokenizer.pkl
```

## Generate levels from text-to-level model

To generate random levels (not based on text embeddings), use this command:
```
python run_diffusion.py --model_path conditional-model --num_samples 100 --text_conditional --save_as_json --output_dir "conditional_model_unconditional_samples"
```
Captions will be automatically assigned to the levels, and you can browse that data with this command:
```
python ascii_data_browser.py conditional_model_unconditional_samples\all_levels.json
```
But to actually provide captions to guide the level generation, use this command
```
python text_to_level_diffusion.py --model_path conditional-model
```
An easier to use GUI interface will let you select and combine known caption phrases to send to the model.
```
python interactive_tile_level_generator.py SMB1_LevelsAndCaptions.json conditional-model
```
Interactively evolve level scenes in the latent space of the conditional model:
```
python evolve_interactive_conditional_diffusion.py --model_path conditional-model
```

## Evaluate caption adherence of text-to-level model

You can evaluate the final model's ability to adhere to input captions with this command:
```
python evaluate_caption_adherence.py --model_path conditional-model --save_as_json
```
You can also evaluate the how caption adherence changed during training with respect to the training set:
```
python evaluate_caption_adherence.py --model_path conditional-model --save_as_json --compare_checkpoints
```
However, it is easy to match the captions used during training. You can evaluate the how caption adherence changed during training with respect to a previously unseen validation set too:
```
python evaluate_caption_adherence.py --model_path conditional-model --save_as_json --compare_checkpoints --json SMB1_ValidationCaptions.json 
```

## Train unconditional diffusion model

To train an unconditional diffusion model without any text embeddings, run this command:
```
python train_diffusion.py --augment --output_dir "unconditional-model" --num_epochs 200
```

## Generate levels from unconditional model

Run trained unconditional diffusion model and save 100 random levels to json
```
python run_diffusion.py --model_path unconditional-model --num_samples 100 --save_as_json --output_dir "unconditional_model_samples"
```
View the saved levels in the data browser
```
python ascii_data_browser.py unconditional_model_samples\all_levels.json
```
Interactively evolve level scenes in the latent space of the unconditional model:
```
python evolve_interactive_unconditional_diffusion.py --model_path unconditional-model
```

## Train Generative Adversarial Network (GAN) model

GANs are an older technology, but they can also be trained to generate levels:
```
python train_wgan.py --augment --num_epochs 5000 --nz 32
```

## Generate levels from GAN

Create samples from final GAN with this command
```
python run_wgan.py --model_path wgan-output\final_models\generator.pth --num_samples 100 --output_dir wgan_samples --save_as_json
```
View the saved levels in the data browser
```
python ascii_data_browser.py wgan_samples\all_levels.json
```
Interactively evolve level scenes in the latent space of the GAN model:
```
python evolve_interactive_wgan.py --model_path wgan-output\final_models\generator.pth
```
