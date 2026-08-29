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
The process of assigning captions to scenes using LLMs is similar to what is possible with Mega Man Maker scenes,
as described [HERE](../Game_MMLV/README.md). Here is how you would assign captions to Mario Maker 2 scenes
using `qwen3.5:9b`:
```
python llm_ascii_to_caption.py --levels Game_MM2\DATA\MM2_LevelsAndCaptions-regular.json --game MM2 --llm ollama --model qwen3.5:9b --output Game_MM2\DATA\MM2_LevelsAndCaptions-llm.json --num_captions 5
```
If you have access to several machines, each running their own local LLM, you can split a single captioning run across them with the `--shard-index` and `--shard-count` parameters. Every machine runs the exact same command, differing only in `--shard-index`. For example, to split the `qwen3.5:9b` run above across 3 machines, you'd run this on the first machine:
```
python llm_ascii_to_caption.py --levels Game_MM2\DATA\MM2_LevelsAndCaptions-regular.json --game MM2 --llm ollama --model qwen3.5:9b --output Game_MM2\DATA\MM2_LevelsAndCaptions-llm.json --num_captions 5 --shard-index 0 --shard-count 3
```
and separate runs with `--shard-index` values of 1 and 2 on different machines. Each machine only captions the scenes assigned to its shard, and each writes its own checkpoint file: `MM2_LevelsAndCaptions-llm.shard0of3.jsonl`, `MM2_LevelsAndCaptions-llm.shard1of3.jsonl`, and `MM2_LevelsAndCaptions-llm.shard2of3.jsonl`. Crash recovery works per shard, so any one machine can be resumed independently of the others.

Once every shard has finished, gather the three checkpoint files into one folder and run `merge_shards.py`, pointing `--shard-count` at the same number you sharded with:
```
python merge_shards.py --output Game_MM2\DATA\MM2_LevelsAndCaptions-llm.json --shard-count 3
```
This reconstructs the shard filenames the same way `llm_ascii_to_caption.py` named them, stitches the scenes back together in their original order, and writes the complete dataset to `Game_MM2\DATA\MM2_LevelsAndCaptions-llm.json`.

Splitting the work with `--shard-index`/`--shard-count` is simple, but it does mean picking a shard count up front and manually merging the results afterward. You can also make multiple machines manage the work on their own and assembly the final file automatically by using `caption_coordinator.py` and `caption_worker.py` instead. In the example below, we will actually collect captions from two different LLMs at the same time.

Start the coordinator once, on any one machine the others can reach over the network, listing all LLMs that will be used:
```
python caption_coordinator.py --levels Game_MM2\DATA\MM2_LevelsAndCaptions-regular.json --game MM2 --output Game_MM2\DATA\MM2_LevelsAndCaptions-llm.json --num_captions 5 --model ollama:qwen3.5:9b --model ollama:gemma4:12b
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

No matter how you choose to create your captioned dataset, you should end up with a file `Game_MM2\DATA\MM2_LevelsAndCaptions-llm.json`. This file needs a train/validation/test split before you can train with `train-diffusion.bat`:
```
python split_data.py --json_file Game_MM2\DATA\MM2_LevelsAndCaptions-llm.json --train_pct 0.9 --val_pct 0.05 --test_pct 0.05 --seed 42 --game MM2
```
Once the data is split, you can use `train-diffusion.bat` like usual:
```
train-diffusion.bat 0 MM2 llm MM2 CLIP single none 0 500 gemma4:12b_captions qwen3.5:9b_captions
```
The dataset gets even more interesting when you add captions generated from commercial LLMs. However, using
a commercial LLM means you have to provide the appropriate API key first.
