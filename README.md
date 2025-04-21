# SDMario

Generate Mario level scenes with Stable Diffusion

## Create data

This repository can be checked out with this command:
```
git clone https://github.com/schrum2/SDMario.git
```
You will also need to check out level data from [TheVGLC](https://github.com/TheVGLC/TheVGLC) to create the training set:
```
git clone https://github.com/TheVGLC/TheVGLC.git
```
Both of these directories should be in the same parent directory. Next, enter the `SDMario` repository.
```
cd SDMario
```
Before running any code, install all requirements with pip:
```
pip install -r requirements.txt
```




CREATE DIFFUSION DATASETS

Extract a json data set of 16 by 16 level scenes from the VGLC data for Super Mario Bros with this command:
```
python create_level_json_data.py --output "SMB1_Levels.json"
```
Create captions for all level scenes in the data set with this command:
```
python create_ascii_captions.py --dataset SMB1_Levels.json --output SMB1_LevelsAndCaptions.json
```
Now you can browse the level scenes and their captions with this command:
```
python ascii_data_browser.py SMB1_LevelsAndCaptions.json 
```
Make a separate validate set of captions with this command:
```
python validation_captions.py --save_file "SMB1_ValidationCaptions.json"
```




TEXT TO LEVEL MODEL


Create a tokenizer for this data with this command:
```
python tokenizer.py save
```
Now, masked language modeling will be used to pre-train the text embedding model.
```
python train_mlm.py --epochs 300
```
A report evaluating the accuracy of the model on the training data is provided after training, but you can repeat a similar evaluation with this command:
```
python masked_token_prediction.py --model_path mlm
```





To train an unconditional diffusion model without any text embeddings, run this command:
```
python train_diffusion.py --augment --output_dir "unconditional-model" --num_epochs 200
```
Run trained diffusion model and save 100 random levels to json
```
python run_diffusion.py --model_path unconditional-model --num_samples 100 --save_as_json --output_dir "unconditional_model_samples"
```
View the saved levels in the data browser
```
python ascii_data_browser.py unconditional_model_samples\all_levels.json
```
Evolve level scenes in the latent space of the model:
```
python evolve_unconditional_diffusion.py --model_path unconditional-model
```







Train a diffusion model conditioned on text embeddings from the descriptive captions:
```
python train_diffusion.py --augment --text_conditional --output_dir "conditional-model" --num_epochs 200
```
To generate random levels (based on random text embeddings), use this command:
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
An easier to use GUI interface will let you select known caption phrases to send to the model.
```
python interactive_tile_level_generator.py SMB1_LevelsAndCaptions.json conditional-model
```
Evaluate the model's ability to adhere to input captions:
```
python evaluate_caption_adherence.py --model_path conditional-model --save_as_json
```
Evaluate the how caption adherence changed during training with respect to the training set:
```
python evaluate_caption_adherence.py --model_path conditional-model --save_as_json --compare_checkpoints --output_dir text_to_level_training
```
Evaluate the how caption adherence changed during training with respect to a validation set:
```
python evaluate_caption_adherence.py --model_path conditional-model --save_as_json --compare_checkpoints --json SMB1_ValidationCaptions.json --output_dir text_to_level_validation
```






Train a GAN
```
python train_wgan.py --augment --num_epochs 5000 --nz 32
```
Create samples from final GAN
```
python run_wgan.py --model_path wgan-output\final_models\generator.pth --num_samples 100 --output_dir wgan_samples --save_as_json
```
View the saved levels in the data browser
```
python ascii_data_browser.py wgan_samples\all_levels.json
```
