# SDMario

Generate Mario level scenes with Stable Diffusion

## Create data

?? Tell how to create level scenes from VGLC? ??

The directory `SMB1` contains level scenes from Super Mario Bros derived from the VGLC. The following command automatically captions each scene and saves the captions in `metadata.jsonl`:
```
python create_level_captions.py SMB1 mario_elements SMB1\\metadata.jsonl
```
Once the captions have been created, you can train the LoRA model. Use the following command:
```
python train_sd15_lora.py -t SMB1 -o SMB1_LoRA -r 256 -s mario
```
Note that the resolution is set to 256, since that is the size of the training images. 


Change instructions above to use "accelerate"


?? interactive plot does not work?
C:\Users\schrum2\Documents\GitHub\SDTinker\train_sd15_lora.py:89: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
?? This command also creates a GUI to plot the change in loss over time. ??

