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
From here, extract an image dataset of level scenes from the level images in TheVGLC:
```
python create_level_squares.py "..\\TheVGLC\\Super Mario Bros\\Original" SMB1 256
```
The 256 parameter slides the window across one level screen at a time. The directory `SMB1` now contains level scenes from Super Mario Bros derived from TheVGLC. The following command automatically captions each scene and saves the captions in `metadata.jsonl`:
```
python create_level_captions.py SMB1 mario_elements SMB1\\metadata.jsonl
```
Once the captions have been created, you can browse them using a GUI with this command:
```
python data_browser.py SMB1\\metadata.jsonl
```
These captions are used to train the LoRA model with this command:
```
python train_sd15_lora.py -t SMB1 -o SMB1_LoRA -r 256 -s mario --plot_loss
```
Note that the resolution is set to 256, since that is the size of the training images. 


Change instructions above to use "accelerate"


?? interactive plot does not work?
C:\Users\schrum2\Documents\GitHub\SDTinker\train_sd15_lora.py:89: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
?? This command also creates a GUI to plot the change in loss over time. ??

