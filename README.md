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
python mlm_training.py --model transformer --epochs 3000
```




TEST trained model? Currently just uses original training data, no test set
```
python .\masked_token_prediction.py --model_file .\mlm_transformer.pth
```








Train diffusion model?
```
python level_diffusion_model.py --json_path path/to/your/data.json --tokenizer_path path/to/tokenizer.json --output_dir ./trained_model --num_tiles 15 --level_size 16 --batch_size 32 --augment --num_train_steps 100000 --use_ema
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