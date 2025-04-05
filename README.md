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




STEPS TO TRAIN A DISCRETE DIFFUSION MODEL:

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
python masked_token_prediction.py --model_file "mlm\\mlm_transformer.pth"
```





To train an unconditional diffusion model without any text embeddings, run this command:
```
python train_diffusion.py --augment --num_epochs 300
```
Run trained diffusion model and save 100 random levels to json
```
python run_diffusion.py --model_path level-diffusion-output --num_samples 100 --save_as_json
```
View the saved levels in the data browser
```
python ascii_data_browser.py generated_levels\\all_levels.json
```








Train a diffusion model conditioned on text embeddings from the descriptive captions:
```
python train_diffusion.py --augment --text_conditional --num_epochs 300
```
To generate random levels (based on random text embeddings), use this command:
```
python run_diffusion.py --model_path level-diffusion-output --num_samples 100 --text_conditional
```
But to actually provide captions to guide the level generation, use this command
```
TODO
```









STEPS BELOW CREATE IMAGES TO TRAIN A STABLE DIFFUSION LORA:

From here, extract an image dataset of level scenes from the level images in TheVGLC:
```
python create_level_squares.py "..\\TheVGLC\\Super Mario Bros\\Original" SMB1
```
The directory `SMB1` now contains level scenes from Super Mario Bros derived from TheVGLC. The following command automatically captions each scene and saves the captions in `metadata.jsonl`:
```
python create_level_captions.py SMB1 mario_elements SMB1\\metadata.jsonl
```
Once the captions have been created, you can browse them using a GUI with this command:
```
python data_browser.py SMB1\\metadata.jsonl
```
Before training, some of the images in `SMB1` need to be deleted, because they contain sprites that are not captioned. Run this command:
```
python filter_training_data.py
```
The remaining data and captions are used to train the LoRA model with this command:
```
python train_sd15_lora.py -t SMB1 -o SMB1_LoRA -r 256 -s mario --plot_loss
```
Note that the resolution is set to 256, since that is the size of the training images. This will take some time to train, but once it is done, you can interactively generate level scenes using the GUI produced by this command:
```
python interactive_level_generator.py SMB1\\metadata.jsonl <safetensors file>
```
You will have to supply the path to the trained safetensors LoRA model. Or you could simply leave these parameters out and use the buttons in the GUI to load the needed files.