# Training models

This document will walk you though all aspects of training diffusion models to generate Mario levels, just like we did. This document will focus heavily on recreating the results from our paper using various batch files, but if you need to run any of the code in a slightly different manner, then you will want to look inside these batch files, or refer to some of the instructions in the original [README](README.md).

## Create Datasets

All the datasets you need are already in the directory named `datasets`, so you can feel free to skip this section. If you choose to go through these steps, then the content in the `datasets` directory should be overwritten with files that are identical to what is already there.

First, you will need to check out 
[my forked copy of TheVGLC](https://github.com/schrum2/TheVGLC). Note that the following command should be executed in the parent directory of the `MarioDiffusion` repository so that the directories for `MarioDiffusion` and `TheVGLC` are next to each other in the same directory:
```
git clone https://github.com/schrum2/TheVGLC.git
```
Once you have my version of `TheVGLC` and `MarioDiffusion`, go into the `batch` sub-directory in the `MarioDiffusion` repo.
```
cd MarioDiffusion
cd batch
```
Next, run a batch file to create datasets from the VGLC data. This batch file call will create sets of 16x16 level scenes of both SMB1 and SMB2 (Japan), as well as a combination of both. Afterwards, it will create captions for all 3 datasets, tokenizers for the data, random test captions for later evaluation, and finally splits the data into training, validation, and testing json files. These files will overwrite the files already in the repo, but they should be identical.
Run this command:
```
Mar1and2-data.bat
```
Now you can browse level scenes and their captions with a command like this (the json file can be replaced by any levels and captions json file in datasets):
```
python ascii_data_browser.py datasets\Mar1and2_LevelsAndCaptions-regular.json 
```
This is not required, but will give you insight into the data.

## 

