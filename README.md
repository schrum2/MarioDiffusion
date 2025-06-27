# Mario Diffusion

Generate Mario level scenes with a diffusion model conditioned on text input.

## Set up the repository

This repository can be checked out with this command:
```
git clone https://github.com/schrum2/MarioDiffusion.git
```
You will also need to check out level data from [My forked copy of TheVGLC](https://github.com/schrum2/TheVGLC) to create the training dataset:
```
git clone https://github.com/schrum2/TheVGLC.git
```
Both of these directories should be in the same parent directory. Next, enter the `MarioDiffusion` repository.
```
cd MarioDiffusion
```
Before running any code, install all requirements with pip:
```
pip install -r requirements.txt
```

## Preview of final results

Following the instructions below will lead you through training your own diffusion model to create Mario levels. There is also a cool GUI you can work with to build larger levels out of diffusion-generated scenes. However, if you want to skip past all of that and just see some results from a pre-trained diffusion model now, run the following command:
```
python .\text_to_level_diffusion.py --model_path "schrum2/MarioDiffusion-MLM-regular0"
```
This will download one of the models from our paper: `MLM-regular`. The model comes from [this Hugging Face repo](https://huggingface.co/schrum2/MarioDiffusion-MLM-regular0). Once it downloads, you will be asked to enter a caption. Try this:
```
full floor. one enemy. a few question blocks. one platform. one pipe. one loose block.
```
For the rest of the prompts, if you simply press enter, it will skip thorugh the default values. Eventually, a level scene will pop up. Congratulations! You've generated your first Mario level scene with one of our diffusion models. Please browse through the instructions below or read our paper to learn more about how our models work. A full list of Hugging Face models you can download are available here:

TODO

## Create datasets

This batch file call will create sets of 16x16 level scenes of both SMB1 and SMB2 (Japan), as well as a combination of both. Afterwards, it will create captions for all 3 datasets, tokenizers for the data, random test captions for later evaluation, and finally splits the data into training, validation, and testing json files. Run these commands:
```
cd batch
Mar1and2-data.bat
```
Now you can browse level scenes and their captions with a command like this (the json file can be replaced by any levels and captions json file in datasets):
```
python ascii_data_browser.py datasets\Mar1and2_LevelsAndCaptions-regular.json 
```

## Complete training and evaluation sequence

The next two sections go into detail on training both the text encoder and the diffusion model, but if you want to train the whole thing all at once and use default settings from our paper, we have some batch files you can use. Be forewarned that after training, these batch files will also embark on a somewhat lengthy data collection process used to evaluate the models, so if you just want to train a model and then play with it yourself, you might want to skip to the more specific instructions below. If you want to use these batch files, you will need to be in the actual batch directory first:
```
cd batch
```
Once here, you can train both a text encoder and its corresponding diffusion model back to back with a single command like this:
```
train-conditional.bat 0 Mar1and2 regular 
```
The `0` is an experiment number which can be replaced with any integer. Both `Mar1and2` and `regular` are referring to portions of the dataset file names that will be used for training, though they also indicate some settings for the model. For example, you can switch `regular` to `absence` and a different style of captions will be used for training. If you switch it to `negative` then negative guidance will be used during training, allowing for negative prompts during inference. If you know you want to repeat an experiment multiple times and train multiple copies of the same model, then you can use this command:
```
batch_runner.bat train-conditional.bat 0 4 Mar1and2 regular
```
This trains models for experiment numbers 0 through 4 in sequence. Also, the primary focus of our work is on training diffusion models that use simple text encoders, but our paper also compares against models using pretrained sentence transformers. They are trained with a different batch file. Here is an example:
```
train-conditional-pre.bat 0 Mar1and2 regular MiniLM split
```
This command trains one diffusion model that uses `MiniLM` as its text model, and the `split` parameter means that individual phrases from the Mario captions each get their own embedding vector. You can simply leave the `split` out to embed each caption with a single vector, and you can also swap `MiniLM` with `GTE`, which is a larger embedding model. It takes longer to train, and is not really worth the extra time, but you are welcome to experiment. The `train-conditional-pre.bat` file can also be used with `batch_runner.bat train-conditional.bat` in a similar way:
```
batch_runner.bat train-conditional-pre.bat 0 4 Mar1and2 regular MiniLM split
```
Now, if you just want to train a model step by step, look at the next sections instead.

## Train text encoder

Masked language modeling is used to train the text embedding model. Use whatever dataset you like with an appropriate tokenizer. It is reccomended to supply the validation and test datasets of the same type as well, though it is optional, and only used for evaluation.
```
python train_mlm.py --epochs 300 --save_checkpoints --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --test_json datasets\Mar1and2_LevelsAndCaptions-regular-test.json --pkl datasets\Mar1and2_Tokenizer-regular.pkl --output_dir Mar1and2-MLM-regular0 --seed 0
```
A report evaluating the accuracy of the final model on the training data is provided after training, but you can repeat a similar evaluation with this command:
```
python evaluate_masked_token_prediction.py --model_path Mar1and2-MLM-regular0 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json
```
You can also see how the accuracy on the training set changes throughout training by evaluating all checkpoints with this command:
```
python evaluate_masked_token_prediction.py --model_path Mar1and2-MLM-regular0 --compare_checkpoints --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json
```
To see accuracy on the validation set over time instead, run this command:
```
python evaluate_masked_token_prediction.py --model_path Mar1and2-MLM-regular0 --compare_checkpoints --json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json
```

## Train text-conditional diffusion model

Now that the text embedding model is ready, train a diffusion model conditioned on text embeddings from the descriptive captions. Note that this can take a while. We used relatively modest consumer GPUs, so our models took about 12 hours to train. However, you can lower the number of epochs to 300 or even 200 and still get decent results:
```
python train_diffusion.py --save_image_epochs 20 --augment --text_conditional --output_dir Mar1and2-conditional-regular0 --num_epochs 500 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --pkl datasets\Mar1and2_Tokenizer-regular.pkl --mlm_model_dir Mar1and2-MLM-regular0 --plot_validation_caption_score --seed 0 
```
Another trick if you care more about speed than seeing intermediate results is to set `--save_image_epochs` to a large number (larger than the number of epochs), like this
```
python train_diffusion.py --save_image_epochs 1000 --augment --text_conditional --output_dir Mar1and2-conditional-regular0 --num_epochs 500 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --pkl datasets\Mar1and2_Tokenizer-regular.pkl --mlm_model_dir Mar1and2-MLM-regular0 --plot_validation_caption_score --seed 0 
```
You can also train with negative prompting by adding an additional flag like this
```
python train_diffusion.py --save_image_epochs 20 --augment --text_conditional --output_dir Mar1and2-conditional-negative0 --num_epochs 500 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --pkl datasets\Mar1and2_Tokenizer-regular.pkl --mlm_model_dir Mar1and2-MLM-regular0 --plot_validation_caption_score --seed 0 --negative_prompt_training
```

## Generate levels from text-conditional diffusion model

To generate unconditional levels (not based on text embeddings), use this batch file:
```
batch\run_diffusion_multi.bat Mar1and2-conditional-regular0 regular Mar1and2 text
```
This batch file automatically creates 2 different sets of 100 samples, one set that is 16 blocks wide, and another that is 128 blocks wide. If you'd like to run just one of these commands, or customize the output further, you can with this command:
```
python run_diffusion.py --model_path Mar1and2-conditional-regular0 --num_samples 100 --text_conditional --save_as_json --output_dir Mar1and2-conditional-regular0-unconditional-samples --level_width 16
```
Captions will be automatically assigned to the levels, and you can browse that data with this command:
```
python ascii_data_browser.py Mar1and2-conditional-regular0-unconditional-samples\all_levels.json
```
But to actually provide captions to guide the level generation, use this command
```
python text_to_level_diffusion.py --model_path Mar1and2-conditional-regular0
```
An easier-to-use GUI interface will let you select and combine known caption phrases to send to the model. Note that the selection of known phrases needs to come from the dataset you trained on.
```
python interactive_tile_level_generator.py --model_path Mar1and2-conditional-regular0 --load_data datasets/Mar1and2_LevelsAndCaptions-regular.json --tileset "..\TheVGLC\Super Mario Bros\smb.json" --game Mario
```
Interactively evolve level scenes in the latent space of the conditional model:
```
python evolve_interactive_conditional_diffusion.py --model_path Mar1and2-conditional-regular0
```
Automatically evolve level scenes in the latent space of the model (must put a caption into the quotations ex "full floor. one enemy."):
```
python evolve_automatic.py --model_path Mar1and2-conditional-regular0 --target_caption " "
```

## Evaluate caption adherence of text-conditional diffusion model

You can evaluate the final model's ability to adhere to input captions with this command:
```
python evaluate_caption_adherence.py --model_path Mar1and2-conditional-regular0 --save_as_json --json datasets\Mar1and2_LevelsAndCaptions-regular.json --output_dir text-to-level-final
```
You can also evaluate the how caption adherence changed during training with respect to the testing set:
```
python evaluate_caption_adherence.py --model_path Mar1and2-conditional-regular0 --save_as_json --json datasets\Mar1and2_LevelsAndCaptions-regular-test.json --compare_checkpoints 
```
However, it is easy to match the captions used during training. You can evaluate the how caption adherence changed during training with respect to a previously unseen randomly generated captions too:
```
python evaluate_caption_adherence.py --model_path Mar1and2-conditional-regular0 --save_as_json --json datasets\Mar1and2_RandomTest-regular.json --compare_checkpoints 
```
If you'd like to do all 3 of these commands at once (as well as automatically generate example level samples), you can do so by running the batch file like this:
```
batch\evaluate_caption_adherence_multi.bat Mar1and2-conditional-regular0 regular Mar1and2
```

## Train unconditional diffusion model

To train an unconditional diffusion model without any text embeddings, run this command:
```
python train_diffusion.py --augment --output_dir Mar1and2-unconditional0 --num_epochs 500 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --seed 0 
```
You can also use this batch file (it also 100 short and 100 long samples from the model once it's trained):
```
cd batch
train-unconditional.bat 0 Mar1and2 
```

## Generate levels from unconditional model

Just like with the text conditional model, you can get level samples from the batch file or a seperate command. The batch file still gets 2 sets of 100 samples, but the arguments are a little different
```
batch\run_diffusion_multi.bat Mar1and2-unconditional0 regular Mar1and2
```
As with before, to get more control, you can simply run this once from the command line
```
python run_diffusion.py --model_path Mar1and2-unconditional0 --num_samples 100 --save_as_json --output_dir Mar1and2-unconditional0-unconditional-samples --level_width 16
```
View the saved levels in the data browser
```
python ascii_data_browser.py Mar1and2-unconditional0-unconditional-samples\all_levels.json
```
Interactively evolve level scenes in the latent space of the unconditional model:
```
python evolve_interactive_unconditional_diffusion.py --model_path Mar1and2-unconditional0
```

## Train Generative Adversarial Network (GAN) model

GANs are an older technology, but they can also be trained to generate levels:
```
python train_wgan.py --augment --json datasets\Mar1and2_LevelsAndCaptions-regular.json --num_epochs 5000 --nz 32 --output_dir Mar1and2-wgan0 --seed 0
```
Just like with the diffusion model, you can save a little bit of time by cutting out intermediate results like this
```
python train_wgan.py --augment --json datasets\Mar1and2_LevelsAndCaptions-regular.json --num_epochs 5000 --nz 32 --output_dir Mar1and2-wgan0 --seed 0 --save_image_epochs 10000
```
You can also use the batch file instead (this will also generate levels with the wgan):
```
cd batch
train-wgan.bat 0 Mar1and2
```

## Generate levels from GAN

Create samples from the final GAN with this command (assuming the batch file hasn't already)
```
python run_wgan.py --model_path Mar1and2-wgan0\final_models\generator.pth" --num_samples 100 --output_dir Mar1and2-wgan0-samples --save_as_json
```
View the saved levels in the data browser
```
python ascii_data_browser.py wgan_samples\all_levels.json
```
Interactively evolve level scenes in the latent space of the GAN model:
```
python evolve_interactive_wgan.py --model_path Mar1and2-wgan0\final_models\generator.pth
```

## Train Five Dollar Model (FDM)
The five-dollar-model is a lightweight feedforward network that trains fast, but has a pretty small maximum performance. They can be trained with a call to the batch file, which will run metrics for you
```
cd batch
train-fdm.bat 0 Mar1and2 regular MiniLM
```
Alternatively, it can be trained individually like so
```
python train_fdm.py --augment --output_dir Mar1and2-fdm-MiniLM-regular0 --num_epochs 100 --json datasets\Mar1and2_LevelsAndCaptions-regular-train.json --val_json datasets\Mar1and2_LevelsAndCaptions-regular-validate.json --pretrained_language_model sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --plot_validation_caption_score --embedding_dim 384 --seed 0
```

## Generate levels from FDM

Create samples from an FDM with this command
```
python text_to_level_fdm.py --model_path Mar1and2-fdm-MiniLM-regular0
```


## Generating MarioGPT data for comparison

Most of the MarioGPT data is taken care of in this batch file, which can be run like this
```
cd batch
MarioGPT-data.bat
```
This batch file generates 96 levels of size 128 using MarioGPT, stores, pads and captions them in the same format as our unconditional models, and then runs metrics on both sliced 16x16 level samples, as well as the full 16x128 generated levels.  

If you'd like to do each of these steps seperatly, that can be done with this series of commands:

First, the level generation can be done with this command, which saves generated levels in a new folder called MarioGPT_Levels, in both text and image format.
```
python run_gpt2.py --output_dir "MarioGPT_Levels" --num_collumns 128
```
Afterwards, this command will take those levels, pad them, and store them in new files in the datasets directory. (The stride variable controls how long individual segments are, the batch file runs this twice to get levels of length 128 and 16)
```
python create_level_json_data.py --output "datasets\\MarioGPT_Levels.json" --levels "MarioGPT_Levels\levels" --stride 16
```
Afterwards, we use this command to give captions to these levels
```
python create_ascii_captions.py --dataset "datasets\\MarioGPT_Levels.json" --output "datasets\\MarioGPT_LevelsAndCaptions-regular.json"
```
And then, lastly, we can use this command to get metrics on the generated levels
```
python calculate_gpt2_metrics.py --generated_levels "datasets\\MarioGPT_LevelsAndCaptions-regular.json" --training_levels "datasets\\Mar1and2_LevelsAndCaptions-regular.json" --output_dir "MarioGPT_metrics"
```

## Comparing model results

Exploration of the time taken to train models, as well as the time to the best epoch can be accomplished by running the following batch file:
```
calculate_runtimes.bat
```
This batch file calls the following batch file to complete all time calculations on every model.
```
calculate_times.bat
```
The output produced above is then used as input by the next script, which aggregates results for plotting.
```
python evaluate_execution_time.py
```
The aggregated results are used to plot the following two figures:

1. A bar plot of mean grouped runtimes for each model with standard error from its individual times:
```
python visualize_best_model_stats.py --input training_runtimes\\mean_grouped_runtimes_plus_best.json --output total_time_with_std_err.pdf --plot_type bar --y_axis "group" --x_axis "mean" --x_axis_label "Hours to Train" --convert_time_to_hours --stacked_bar_for_mlm 
```
2. A box plot of individual times by each model:
```
python visualize_best_model_stats.py --input mean_grouped_runtimes.json --output mean_grouped_runtime_box_plot.pdf --plot_type box --x_axis "group" --y_axis "individual_times" --x_axis_label "Models" --y_axis_label "Hours to Train" --convert_time_to_hours --x_tick_rotation 45
```
Lastly, data from best_model_info.json from each model are aggregated for visualization:
```
python best_model_statistics.py
```
The output from this script is used to plot the following four graphs:
1. A box plot for the best epoch of each model type:
```
python visualize_best_model_stats.py --input best_model_info_20250618_162147.json --output best_epoch_box_plot.pdf --plot_type box --x_axis "group" --y_axis "best_epoch" --x_axis_label "Models" --y_axis_label "Best Epoch" --x_tick_rotation 45 
```
2. A violin plot for the best epoch of each model type:
```
python visualize_best_model_stats.py --input best_model_info_20250618_162147.json --output best_epoch_violin_plot.pdf --plot_type violin --x_axis "group" --y_axis "best_epoch" --y_axis_label "Best Epoch" --x_tick_rotation 45 
```
3. A bar plot for the best epoch of each model type:
```
python visualize_best_model_stats.py --input best_model_info_20250618_162147.json --output best_epoch_bar_plot.pdf --plot_type bar --y_axis "group" --x_axis "best_epoch" --x_axis_label "Best Epoch" --x_markers_on_bar_plot 
```
4. A scatter plot for that compares the best caption score by the best epoch:
```
python visualize_best_model_stats.py --input best_model_info_20250618_162147.json --output best_epoch_v_best_caption_score_scatter_plot.pdf --plot_type scatter --y_axis "best_caption_score" --x_axis "best_epoch" --y_axis_label "Best Caption Score" --x_axis_label "Best Epoch" 
```

Evaluating A* Solvability for each model using up to 100 samples from all_levels.json from each model. This returns astar_result_overall_averages.json, the average across all averages of metrics returned from all A* runs on tested levels. Returns from the following batch file are automatically plotted in plot_metrics.bat (described in the next section.)
```
evaluate-solvability.bat
```

After running this batch file, you can plot these results on their own with the following command:
```
python evaluate_models.py --plot_file astar_result_overall_averages.json --modes real random short --metric "beaten" --plot_label "Percent Beatable Levels" --save
```

Average minimum edit distance (amed) calculates the edit distance for each level in a levelset against a levelset. We calculate amed self, where the min edit distance is calculated for each level against the remaining levels in the set, and amed real, where the min edit distance is calculated for each level in a levelset against the entire real levelset that was used to generate the level.

All amed plots and calculations - as well as broken feature generatsion plots - can be run like this
```
cd batch
plot_metrics.bat
```
This batch file will generate the following plots at once

```
python evaluate_models.py --modes real random short real_full --full_metrics --metric average_min_edit_distance_from_real --plot_label "Edit Distance" --save --output_name "AMED-REAL_real(full)_real(100)_random_unconditional" --loc "best" --legend_cols 1 --errorbar
```
This saves an amed real plot between generated levels against the real dataset
```
python evaluate_metrics.py --real_data --model_path None
```
Resample the original dataset that levels were created on to 100 samples for a fair comparison
```
python evaluate_models.py --modes real random short real_full --full_metrics --metric average_min_edit_distance --plot_label "Edit Distance" --save --output_name "AMED-SELF_real(full)_real(100)_random_unconditional" --loc "lower right" --bbox 1.0 0.1 --errorbar
```
Plots comparison of the amed between generated levels themselves against different models
```
python evaluate_models.py --modes real random short real_full --full_metrics --metric broken_pipes_percentage_in_dataset --plot_label "Percent Broken Pipes" --save --output_name "BPPDataset_real(full)_real(100)_random_unconditional" --loc "lower right" --legend_cols 2 --errorbar
```
Plots and compares broken pipes as a percentage of the full generated dataset
```
python evaluate_models.py --modes real random short real_full --full_metrics --metric broken_pipes_percentage_of_pipes --plot_label "Percent Broken Pipes" --save --output_name "BPPPipes_real(full)_real(100)_random_unconditional" --loc "lower right" --legend_cols 2 --errorbar
```
Plots and compares broken pipes as a percentage of total pipe mentions
```
python evaluate_models.py --modes real random short real_full --full_metrics --metric broken_cannons_percentage_in_dataset --plot_label "Percent Broken Cannons" --save --output_name "BCPDataset_real(full)_real(100)_random_unconditional" --loc "lower right" --errorbar
```
Plots and compares broken cannons as a percentage of the full generated dataset
```
python evaluate_models.py --modes real random short real_full --full_metrics --metric broken_cannons_percentage_of_cannons --plot_label "Percent Broken Cannons" --save --output_name "BCPCannons_real(full)_real(100)_random_unconditional" --loc "lower right" --errorbar
```
Plots and compares broken cannons as a percentage of total cannon mentions
