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
python create_megaman_json_data.py --levels Game_MMLV\MMLV_Levels --tileset Game_MMLV\MMLV.json --stride_x 16 --stride_y 14 --scan_mode screen_grid --include_moving_ground --output Game_MMLV\DATA\MMLV_Levels.json --max_enemies 8 --min_content_pct 7
```
Notice that we also filter out some of the level samples, either because they contain
mechanics too complicated for us to model with our tileset, or because the quality of the
scene seems poor ... keep in mind that anyone can make a Mega Man Maker level, and not all
are good.

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
python llm_ascii_to_caption.py --levels Game_MMLV\DATA\MMLV_LevelsAndCaptions-regular.json --game MMLV --llm ollama --model qwen3.5:9b --output Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.json --num_captions 5
```
Running this command takes a long time, so you may want to consider various options for breaking up the task. First, be aware that the code saves incremental progress to a .jsonl file matching the name of your intended .json output file. So, for the example above, there is a file named 
`Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.jsonl` during execution. If execution completes successfully, you may want to delete this file.
However, if execution was interrupted and you rerun the same command, and that checkpoint file is still there, you'll be asked whether to resume.

If you have access to several machines, each running their own local LLM, you can split a single captioning run across them with the `--shard-index` and `--shard-count` parameters. Every machine runs the exact same command, differing only in `--shard-index`. For example, to split the `qwen3.5:9b` run above across 3 machines, you'd run this on the first machine:
```
python llm_ascii_to_caption.py --levels Game_MMLV\DATA\MMLV_LevelsAndCaptions-regular.json --game MMLV --llm ollama --model qwen3.5:9b --output Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.json --num_captions 5 --shard-index 0 --shard-count 3
```
and separate runs with `--shard-index` values of 1 and 2 on different machines. Each machine only captions the scenes assigned to its shard, and each writes its own checkpoint file: `MMLV_LevelsAndCaptions-llm.shard0of3.jsonl`, `MMLV_LevelsAndCaptions-llm.shard1of3.jsonl`, and `MMLV_LevelsAndCaptions-llm.shard2of3.jsonl`. Crash recovery works per shard exactly as described above, so any one machine can be resumed independently of the others.

Once every shard has finished (or you just want to check progress on a still-running set of shards), gather the three checkpoint files into one folder and run `merge_shards.py`, pointing `--shard-count` at the same number you sharded with:
```
python merge_shards.py --output Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.json --shard-count 3
```
This reconstructs the shard filenames the same way `llm_ascii_to_caption.py` named them, stitches the scenes back together in their original order, and writes the complete dataset to `Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.json`.

Splitting the work with `--shard-index`/`--shard-count` is simple, but it does mean picking a shard count up front and manually merging the results afterward. You can also make multiple machines manage the work on their own and assembly the final file automatically by using `caption_coordinator.py` and `caption_worker.py` instead. In the example below, we will actually collect captions from two different LLMs at the same time.

Start the coordinator once, on any one machine the others can reach over the network, listing all LLMs that will be used:
```
python caption_coordinator.py --levels Game_MMLV\DATA\MMLV_LevelsAndCaptions-regular.json --game MMLV --output Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.json --num_captions 5 --model ollama:qwen3.5:9b --model ollama:gemma4:12b
```
The coordinator will announce its IP address when it starts, for example, `10.117.56.119`. In that case, you would launch the following two commands on various other machines on the same network:
```
python caption_worker.py --coordinator http://10.117.56.119:8765 --llm ollama --model qwen3.5:9b
```
That is for the machines that run Qwen, and the command below is for the machines that run Gemma:
```
python caption_worker.py --coordinator http://10.117.56.119:8765 --llm ollama --model gemma4:12b
```
Once the coordinator has collected captions for all scenes using all designated LLMs, it will tell the workers that it is done.

No matter how you choose to create your captioned dataset, you should end up with a file `Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.json`. This file needs a train/validation/test split before you can train with `train-diffusion.bat`:
```
python split_data.py --json_file Game_MMLV\DATA\MMLV_LevelsAndCaptions-llm.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MMLV
```
Once the data is split, you can use `train-diffusion.bat` like usual:
```
train-diffusion.bat 0 MMLV llm MMLV CLIP single none 0 500 gemma4:12b_captions qwen3.5:9b_captions deterministic_captions
```
Note that this training command added `deterministic_captions` into the mix along with the LLM-generated captions.
The dataset gets even more interesting when you add captions generated from commercial LLMs. However, using
a commercial LLM means you have to provide the appropriate API key first.













TODO: Talk about API keys for commercial LLMs, but also deal with how big the data is. How to distribute creation of captions.
