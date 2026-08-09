# Mega Man Maker Diffusion

These instructions also deal with Mega Man, but focus on a different data source.
[Mega Man Maker](https://megamanmaker.com/) is a fan-made program for designing and playing
Mega Man levels. Once you have followed instructions here to create datasets based
on Mega Man Maker's MMLV format, you can slightly modify instructions for training
[Mega Man models with VGLC data](../Game_MM/README.md) and apply them here.
As always, you should follow the basic setup instructions in the main [README](../README.md) first.

## Batch Files

Our code was developed on Windows machines, so we have made extensive use of batch files for convenience. However, these will not work on Linux/Mac systems. The Python scripts that are called from these batch files should work on any system, though this has not been fully tested. The instructions below describe how to use the batch files and certain Python scripts, but you can look in the batch files to execute individual commands as needed.

## Mega Man Maker

You need to start by downloading and installing [Mega Man Maker](https://megamanmaker.com/) first.
Some steps of the dataset creation process will download MMLV levels to a directory associated
with Mega Man Maker, so it is important to install the program first. Installing the program
also means it will be possible to play your created levels later as well.

## Create datasets


TODO: batch file that takes care of everything

TODO: Now break it down step by step


## Bulk Download MMLV Levels

Mega Man Maker levels are freely available online in the MMLV format.
You will need many MMLV levels in order to make a suitable training set,
so you can use the following command (from the `Game_MMLV` directory) to download levels in bulk.
```
cd Game_MMLV
python Bulk_Download.py --target 5000
```
