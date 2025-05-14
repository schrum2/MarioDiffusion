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

Extract a json data set of 16 by 16 level scenes from the VGLC data for Super Mario Bros with this command:
```
python create_level_json_data.py --output "SMB1_Levels.json"
```
You can also extract data from Super Mario Bros 2 (Japan) with this command:
```
python create_level_json_data.py --output "SMB2_Levels.json" --levels "..\\TheVGLC\\Super Mario Bros 2 (Japan)\\Processed"
```
Finally, data from the Gameboy game Super Mario Land is extracted with this command:
```
python create_level_json_data.py --output "SML_Levels.json" --levels "..\\TheVGLC\\Super Mario Land\\Processed"
```
Combine the data from the three Mario games into a single dataset:
```
python combine_data.py Mario_Levels.json SMB1_Levels.json SMB2_Levels.json SML_Levels.json
```
These files only contains the level scenes. Create captions for all level scenes with this command:
```
python create_ascii_captions.py --dataset Mario_Levels.json --output Mario_LevelsAndCaptions.json
```
Now you can browse the level scenes and their captions with these commands:
```
python ascii_data_browser.py Mario_LevelsAndCaptions.json 
```
Make a separate validation set of captions with this command. These randomly generated captions are used for validation later (using seed 0 will give you the same validation set used in our experiments) (THIS NEEDS THE TOKENIZER ... WHY?):
```
python create_validation_captions.py --save_file "Mario_ValidationCaptions.json" --seed 0
```

## Can I also get Mega Man Data?

This doesn't work yet
```
python create_level_json_data.py --output "MM_Levels.json" --levels "..\\TheVGLC\\MegaMan"
```

## Train text encoder

First create a tokenizer for the caption data from all Mario games:
```
python tokenizer.py save --json_file Mario_LevelsAndCaptions.json --pkl_file Mario_Tokenizer.pkl
```
Now, masked language modeling will be used to pre-train the text embedding model.
```
python train_mlm.py --epochs 300 --save_checkpoints
python train_mlm.py --epochs 300 --save_checkpoints --json SMB2_LevelsAndCaptions.json --pkl SMB2_Tokenizer.pkl --output_dir mlm2
```
A report evaluating the accuracy of the final model on the training data is provided after training, but you can repeat a similar evaluation with this command:
```
python evaluate_masked_token_prediction.py --model_path mlm
```
You can also see how the accuracy on the training set changes throughout training by evaluating all checkpoints with this command:
```
python evaluate_masked_token_prediction.py --model_path mlm --compare_checkpoints
```
To see accuracy on the validation set over time instead, run this command:
```
python evaluate_masked_token_prediction.py --model_path mlm --compare_checkpoints --json SMB1_ValidationCaptions.json
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
