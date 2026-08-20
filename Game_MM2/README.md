# Mario Maker Diffusion

These instructions deal with Mario levels designed for Mario Maker 1 and 2.

MORE INTRO

Once you have followed instructions here to create datasets based
on Mario Maker levels, 

HOW TO TRAIN

As always, you should follow the basic setup instructions in the main [README](../README.md) first.

## Batch Files

Our code was developed on Windows machines, so we have made extensive use of batch files for convenience. However, these will not work on Linux/Mac systems. The Python scripts that are called from these batch files should work on any system, though this has not been fully tested. The instructions below describe how to use the batch files and certain Python scripts, but you can look in the batch files to execute individual commands as needed.

## TODO

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















Data source, MariOver repo, etc

## Create datasets

We will first retrieve levels from the dataset on Hugging Face.


Basically, run MM2_Batch\MM2-extract.bat, then MM2_Batch\MM2-data.bat

