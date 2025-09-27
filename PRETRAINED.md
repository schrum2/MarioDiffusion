
# Using pretrained models

We have made various models available on Hugging Face for ease of use. Specifically, one diffusion model of each type described in our paper is available. If you want to use the one that we believe is both the best and easiest to use, then stick to `schrum2/MarioDiffusion-MLM-regular0`. The full list of models is available [here](MODELS.md), but here are instructions on how to use them.

## Command line

To download and interact with a model via the command line, run the following command:
```
python .\text_to_level_diffusion.py --model_path "schrum2/MarioDiffusion-MLM-regular0"
```
This will download the `MLM-regular` model from [this Hugging Face repo](https://huggingface.co/schrum2/MarioDiffusion-MLM-regular0). Once it downloads, you will be asked to enter a caption. Try this:
```
full floor. one enemy. a few question blocks. one platform. one pipe. one loose block.
```
For the rest of the prompts, if you simply press enter, it will skip thorugh the default values. Eventually, a level scene will pop up. Congratulations! You've generated your first Mario level scene with one of our diffusion models. You can exit the program by providing an input of 'q' to any of the prompts.

Note that if you use a model trained with absence captions, then more information will be expected in the input caption. For example, you can use the `MLM-absence` model with this command:
```
python .\text_to_level_diffusion.py --model_path "schrum2/MarioDiffusion-MLM-absence0"
```
But the caption corresponding to the one above would be:
```
full floor. one enemy. a few question blocks. one platform. one pipe. one loose block. no ceiling. no upside down pipes. no coin lines. no coins. no towers. no cannons. no ascending staircases. no descending staircases. no rectangular block clusters. no irregular block clusters. no question blocks. no loose blocks.
```
To avoid the need to always provide the absence information, you can use the `--automatic_absence_captions` option like this:
```
python .\text_to_level_diffusion.py --model_path "schrum2/MarioDiffusion-MLM-absence0" --automatic_absence_captions
```
This means that you can provide input captions without any absence phrases, but they will be added automatically to the input to the model.

If you interact with a model that supports negative text guidance, such as `schrum2/MarioDiffusion-MLM-negative0`, then there will be an additional input called 'negative_prompt'. However, we recommend using the `--automatic_negative_captions` option so that this extra prompt goes away and is instead handled automatically. Simply run this command:
```
python .\text_to_level_diffusion.py --model_path "schrum2/MarioDiffusion-MLM-negative0" --automatic_negative_captions
```

Feel free to experiment with the other input prompts. Here is some information on each value:

1. `width`: You can generate longer or shorter levels. The models may have more trouble following your text guidance in this case, but results will be generated.
2. `start_seed` and `end_seed`: If you change these, then be sure to set `start_seed <= end_seed`, as the model will be used to generate multiple levels with different random seeds.
3. `num_inference_steps`: How many times the model is executed. The noise output of the model is subtracted from the input, and the process is repeated with more noise removed each time until a (hopefully) completely denoised result is generated.
4. `guidance_scale`: Influences the balance between unconditional generation and adherence to the provided text guidance when levels are generated.
5. `Do you want to play this level? (y/n)`: This is asked after the level is generated but before you see it. If 'y' is selected, then an A* agent will play the level.

As useful as this tool is, we feel that most will have more fun generating levels with the GUI described next.

## Graphical User Interface

There is also a cool GUI you can work with to build larger levels out of diffusion-generated scenes. Run the following command to load the `MLM-regular` model in the GUI and create new levels interactively (**Note**: sufficiently large screen resolution will be needed to view all GUI elements).

```
python interactive_tile_level_generator.py --model_path schrum2/MarioDiffusion-MLM-regular0 --load_data datasets/Mar1and2_LevelsAndCaptions-regular.json
```

**DETAILS:**
1. Adjust the 'Number of Images' or change the 'Random Seed', and click the 'Generate Image' button to generate new levels. 
2. For more control of the content generated, use the drop down menus on the right side of the GUI. Here you can construct a caption for text guidance by checking boxes. A caption can also be typed directly into the 'Constructed Caption' box.
3. For larger levels, you can adjust the 'Width' and 'Height', but directly generating unusually shaped levels with the diffusion model can lead to weird results. Alternatively, you can compose larger levels from generated content by clicking the 'Add to Level' button under a specific level scene that was made. 
4. Once scenes are added to larger composed levels, click on the thumbnails you would like to delete or rearrange.
5. You may also play or run A* Mario on any generated level or composed larger level, and you can toggle between SNES and NES graphics with the 'Use SNES Graphics' checkbox.
6. Save composed larger levels as ASCII text files by clicking on 'Save Composed Level.'
7. There is an input box for a 'Negative Prompt', but you should not expect this to do anything useful unless you are using a model trained with negative guidance, such as `schrum2/MarioDiffusion-MLM-negative0`. Even then, we recommend simply checking the box for 'Automatic Negative Captions' to assure consistent input.
8. If you are working with a model trained on absence captions, then when launching the GUI, you should specify this with `--load_data datasets/Mar1and2_LevelsAndCaptions-absence.json` so that the phrase options on the right include absence phrases. However, when using models trained on absence captions, we recommend checking the 'Automatic Absence Captions' box.

## Conclusion

We hope these tools for interacting with pre-trained models provide you with a fun way of seeing what is possible with diffusion models. If you would like to train models yourself, then please go to [TRAINING.md](TRAINING.md).