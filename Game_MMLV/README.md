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



Mega Man Maker levels are freely available online in the MMLV format.
You will need many MMLV levels in order to make a suitable training set,
so you can use the following command (from the `Game_MMLV` directory) to download levels in bulk.
```
cd Game_MMLV
python bulk_mmlv_download.py --target 5000
```
This command attempts to download 5000 valid MMLV levels for use as training data.
The files are downloaded to `%LOCALAPPDATA%\MegaMaker\Levels` which is a different
directory for each user of your personal Windows machine. This is the same
directory where Mega Man Maker downloads levels that you want to play from the Mega Man Maker servers.
The process takes a while since the program simply checks sequential level ID numbers,
but levels with some IDs don't exist.

After the levels are downloaded, they need to be converted into 2D ASCII representation, with this command (also in `Game_MMLV`):
```
python bulk_mmlv_to_vglc.py --output MMLV_Levels
```
This format shares some similarities with the VLGC format, but allows many more tile types,
while also lacking some tile types that are present in the `MM-Full` tileset. The MMLV tileset
is in `Game_MMLV/MMLV.json`. The command above saves a txt file into `Game_MMLV/MMLV_Levels`
for each MMLV level downloaded in the previous step. 
The format does not represent the full variety of Mega Man levels, ignoring most
stylistic and artistic distinctions. Some complicated game mechanics that would
be difficult to represent with a purely 2D ASCII representation are also lost.
Each of these files represents a complete
level, so they are broken up into individual scenes using this command, which needs to be run
from the root project directory rather than the `Game_MMLV` subdirectory:
```
python create_megaman_json_data.py --levels Game_MMLV\MMLV_Levels --tileset Game_MMLV\MMLV.json --stride_x 16 --stride_y 14 --scan_mode screen_grid --include_moving_ground --output Game_MMLV\DATA\MMLV_Levels.json --no_traversable_filter --max_enemies 8 --min_content_pct 7
```
Once the level scenes have been created, they can be assigned captions. Given the increased diversity of MMLV levels,
we are generally more interested in the diversity of LLM-assigned captions when it comes to these levels,
but you can still assign deterministic captions with this command:
```
python MM_create_ascii_captions.py --dataset Game_MMLV\DATA\MMLV_Levels.json --tileset Game_MMLV\MMLV.json --output Game_MMLV\DATA\MMLV_LevelsAndCaptions-regular.json
```
