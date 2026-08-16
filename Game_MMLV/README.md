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

## Create datasets: Deterministic Captions

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
python MM_create_ascii_captions.py --dataset Game_MMLV\DATA\MMLV_Levels.json --tileset Game_MMLV\MMLV.json --output Game_MMLV\DATA\MMLV_LevelsAndCaptions-regular.json --caption-mode keyed --caption-key deterministic_captions
```
The use of `--caption-key deterministic_captions` stores the captions differently than in `MM-Simple` and `MM-Full` datasets.
The one deterministic caption is stored in a list under the key `deterministic_captions`, which makes it easier to combine
with LLM-generated captions.

All of the above steps can be carried out by simply executing a single batch file:
```
cd Game_MMLV
cd BATCH
MMLV-data.bat
```
This is a good starting point before generating LLM-based captions for level scenes.

## LLM-Generated Captions

The [README](../Game_MM/README.md) for the `MM-Simple` and `MM-Full` tilesets already discusses
the assigning of LLM-based captions to a dataset, and the procedure is essentially the same
with the MMLV levels. Here is how you would assign captions from `qwen3.5:9b`:
```
python llm_ascii_to_caption.py --levels Game_MMLV\DATA\MMLV_LevelsAndCaptions-regular.json --game MMLV --llm ollama --model qwen3.5:9b --output Game_MM\DATA\MMLV_LevelsAndCaptions-llm.json --num_captions 5
```
And then you can add captions from `gemma4:12b` to the same dataset with this command:
```
python llm_ascii_to_caption.py --levels Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.json --game MMLV --llm ollama --model gemma4:12b --output Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.json --num_captions 5
```
Keep in mind that `train-diffusion.bat` assumes there is a train/validate/test split of the data, so you will need to split the data before using this batch file:
```
python split_data.py --json_file Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MMLV
```
Once the data is split, you can use `train-diffusion.bat` like usual:
```
train-diffusion.bat 0 MMLV llm MMLV CLIP single none 0 gemma4:12b_captions qwen3.5:9b_captions deterministic_captions
```
Note that this training command added `deterministic_captions` into the mix along with the LLM-generated captions.
Still, none of this is much different than the instructions for the `MM-Simple` and `MM-Full` tilesets.
The dataset gets more interesting when you add captions generated from commercial LLMs. However, using
a commercial LLM means you have to provide the appropriate API key first.







TODO: Talk about API keys for commercial LLMs, but also deal with how big the data is. How to distribute creation of captions.