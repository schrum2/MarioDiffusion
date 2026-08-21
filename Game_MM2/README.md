# Mario Maker Diffusion

These instructions deal with Mario levels designed for Mario Maker 1 and 2.

MORE INTRO

Once you have followed instructions here to create datasets based
on Mario Maker levels, 

HOW TO TRAIN

As always, you should follow the basic setup instructions in the main [README](../README.md) first.

## Batch Files

Our code was developed on Windows machines, so we have made extensive use of batch files for convenience. However, these will not work on Linux/Mac systems. The Python scripts that are called from these batch files should work on any system, though this has not been fully tested. The instructions below describe how to use the batch files and certain Python scripts, but you can look in the batch files to execute individual commands as needed.

## Create datasets: Deterministic Captions

The data that we train on comes from Mario Maker 2. GitHub user [TheGreatRambler](https://github.com/TheGreatRambler) 
made a repository [MariOver](https://github.com/TheGreatRambler/MariOver) that allows these levels to be extracted
in a comprehensible format. Furthermore, a large dataset of these levels was stored by TheGreatRambler in this 
[HuggingFace repository](https://huggingface.co/datasets/TheGreatRambler/mm2_level). Our code actually extracts
the levels from this repository and manipulates it into a format suitable for training. The easiest way to start
this process is to run our batch files. First:
```
cd Game_MM2
cd BATCH
MM2-extract.bat 10000 10
```
The 10000 is the number of levels that will be downloaded and processed, and 10 is the minimum number of "likes" 
the levels needs to have received to be considered worthy of processing. Because of the wide range of quality
in Mario Maker levels, we did not think it was appropriate to train on data from all available levels.
This batch file goes through the steps of downloading `.bcd` files of Mario Maker 2 levels into
`Game_MM2\LEVELS\bcd`, and then converting those levels to a `.json` format that is easier to process.
Those `.json` files are stored in `Game_MM2\LEVELS\json` while a visualization of every downloaded level is
saved into `Game_MM2\LEVELS\images`. Finally, the `.json` files are converted into 2D ASCII text grids which
are stored in `Game_MM2\LEVELS\ascii`.

The next major step of the process involved another batch file which extracts scenes from this ASCII representation
and stores them into the `.json` format used for training data with all other games supported by this repo.
The code also assigns simple deterministic captions pointing out the presence and quantity of different
entities, generates random test captions based on this simplistic caption scheme, and creates a train/validate/test
split of the data. To carry out all of these steps, launch this batch file (still inside `Game_MM2\BATCH`):
```
cd Game_MM2
cd BATCH
MM2-data.bat
```
The `.json` files produced by this are in `Game_MM2\DATA`. The collection of all samples is in `MM2_LevelsAndCaptions-regular.json` and the train/validate/test split files have similar names.

## Train diffusion on deterministic captions

Once this dataset is created, commands similar to those used with other games are supported. For example,
here is a command to train an MLM text encoder that is then used to train a diffusion model on the deterministic
captions:
```
cd batch
train-diffusion.bat 0 MM2 regular MM2 MLM
```

## LLM-Generated Captions

Though models can be trained with the deterministic captions, we can also make more complex captions with the aide of LLMs.















NOT TRUE YET. CURRENT INSTRUCTIONS ARE WAY DIFFERENT

The instructions here are somewhat similar to those used with Mega Man Maker, available in this [README](../Game_MMLV/README.md).
