import argparse
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from datetime import datetime
from level_dataset import LevelDataset, visualize_samples
import json
from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from models.fdm_pipeline import FDMPipeline
from level_dataset import visualize_samples, convert_to_level_format, samples_to_scenes
from create_ascii_captions import assign_caption, save_level_data
from LR_create_ascii_captions import assign_caption as lr_assign_caption
from LR_create_ascii_captions import save_level_data as lr_save_level_data
from captions.util import extract_tileset 
from captions.caption_match import compare_captions
from tqdm.auto import tqdm
import util.common_settings as common_settings
from models.fdm_pipeline import FDMPipeline
from util.plotter import Plotter  

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate caption adherence for a pretrained text-conditional diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=common_settings.MARIO_TILE_COUNT, help="Number of tile types")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size") 
        
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--inference_steps", type=int, default=common_settings.NUM_INFERENCE_STEPS, help="Number of denoising steps") # Large reduction from the 500 used during training
    parser.add_argument("--guidance_scale", type=float, default=common_settings.GUIDANCE_SCALE, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--save_as_json", action="store_true", help="Save generated levels as JSON")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted checkpoint comparison run")

    # Used to generate captions when generating images
    parser.add_argument("--tileset", default='..\\TheVGLC\\Super Mario Bros\\smb.json', help="Descriptions of individual tile types")
    #parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--width", type=int, default=common_settings.MARIO_WIDTH, help="Width of the generated levels")
    parser.add_argument("--height", type=int, default=common_settings.MARIO_HEIGHT, help="Height of the generated levels")

    # Output args
    parser.add_argument("--output_dir", type=str, default="text_to_level_results", help="Output directory if not comparing checkpoints (subdir of model directory)")
    parser.add_argument("--save_image_samples", action="store_true", help="Save generated levels in png files")

    parser.add_argument("--compare_checkpoints", action="store_true", default=False, help="Run comparison across all model checkpoints")

    return parser.parse_args()

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"     # Save within the model path directory

    # Based on the number of tiles, decides which game to run
    if args.num_tiles == common_settings.MARIO_TILE_COUNT:
            tileset = '..\\TheVGLC\\Super Mario Bros\\smb.json'
            height = common_settings.MARIO_HEIGHT
            width = common_settings.MARIO_WIDTH
            path_to_json = args.json
    elif args.num_tiles == common_settings.LR_TILE_COUNT:
            tileset = '..\\TheVGLC\\Lode Runner\\Loderunner.json'
            height = common_settings.LR_HEIGHT
            width = common_settings.LR_WIDTH
            path_to_json = "datasets\LR_LevelsAndCaptions-regular.json"


    if not args.compare_checkpoints:
        args.output_dir = os.path.join(args.model_path, args.output_dir)
        # Check if output directory already exists
        if os.path.exists(args.output_dir):
            print(f"Error: Output directory '{args.output_dir}' already exists. Please remove it or specify a different output directory.")
            exit(1)
        # Create output directory
        os.makedirs(args.output_dir)

    _, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset)
        
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    using_unet_pipe = False
    # TODO: This won't work if training terminated early, but there are still valid checkpoints I want to evaluate
    if(os.path.exists(os.path.join(args.model_path, "unet"))):
        #Default to getting the Diffusion pipeline
        using_unet_pipe = True
        pipe = TextConditionalDDPMPipeline.from_pretrained(args.model_path).to(device)
    else:
        #Get the FDM pipeline if "unet" doesn't exist
        pipe = FDMPipeline.from_pretrained(os.path.join(args.model_path, "final-model")).to(device)

    assert(pipe.tokenizer is not None)

    # Initialize dataset
    dataset = LevelDataset(
        json_path=path_to_json,
        tokenizer=None,
        shuffle=False,
        mode="text",
        augment=False,
        num_tiles=args.num_tiles
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    if args.compare_checkpoints:
        scores_by_epoch = track_caption_adherence(args, device, dataloader, id_to_char, char_to_id, tile_descriptors, using_unet_pipe)

    else:
        # Just run on one model and get samples as well
        avg_score, all_samples, all_prompts, _ = calculate_caption_score_and_samples(device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, args.describe_absence, output=False, height=height, width=width)

        print(f"Average caption adherence score: {avg_score:.4f}")
        print(f"Generated {len(all_samples)} level samples")
        
        if args.save_image_samples:
            visualize_samples(all_samples, args.output_dir, prompts=all_prompts)

        if args.save_as_json:
            scenes = samples_to_scenes(all_samples)
            if args.num_tiles == common_settings.MARIO_TILE_COUNT:
                save_level_data(scenes, args.tileset, os.path.join(args.output_dir, "all_levels.json"), False, args.describe_absence, exclude_broken=False, prompts=all_prompts)
            elif args.num_tiles == common_settings.LR_TILE_COUNT:
                tileset = '..\\TheVGLC\\Lode Runner\\Loderunner.json'
                lr_save_level_data(scenes, tileset, os.path.join(args.output_dir, "all_levels.json"), False, args.describe_absence)


def track_caption_adherence(args, device, dataloader, id_to_char, char_to_id, tile_descriptors, using_unet_pipe=True):

    if args.num_tiles == common_settings.MARIO_TILE_COUNT:
            tileset = '..\\TheVGLC\\Super Mario Bros\\smb.json'
            height = common_settings.MARIO_HEIGHT
            width = common_settings.MARIO_WIDTH
            path_to_json = args.json
    elif args.num_tiles == common_settings.LR_TILE_COUNT:
            tileset = '..\\TheVGLC\\Lode Runner\\Loderunner.json'
            height = common_settings.LR_HEIGHT
            width = common_settings.LR_WIDTH
            path_to_json = "datasets\LR_LevelsAndCaptions-regular.json"

    checkpoint_dirs = [
        (int(d.split("-")[-1]), os.path.join(args.model_path, d))
        for d in os.listdir(args.model_path)
        if os.path.isdir(os.path.join(args.model_path, d)) and d.startswith("checkpoint-")
    ]    
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: x[0])
    if os.path.isdir(os.path.join(args.model_path, "unet")):
        checkpoint_dirs.append((checkpoint_dirs[-1][0] + 1, args.model_path))

    # Prepare output paths
    scores_jsonl_path = os.path.join(args.model_path, f"{os.path.basename(path_to_json).split('.')[0]}_scores_by_epoch.jsonl")
    plot_png_path = os.path.join(args.model_path, f"{os.path.basename(path_to_json).split('.')[0]}_caption_scores_plot.png")

    # Handle file existence based on resume flag
    completed_epochs = set()
    if os.path.exists(scores_jsonl_path):
        if args.resume:
            # Create backup files with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            if os.path.exists(scores_jsonl_path):
                backup_jsonl = scores_jsonl_path.replace('.jsonl', f'_backup_{timestamp}.jsonl')
                os.rename(scores_jsonl_path, backup_jsonl)
                # Copy content back to original file
                with open(backup_jsonl, 'r') as src, open(scores_jsonl_path, 'w') as dst:
                    for line in src:
                        entry = json.loads(line)
                        completed_epochs.add(entry['epoch'])
                        dst.write(line)
            if os.path.exists(plot_png_path):
                backup_png = plot_png_path.replace('.png', f'_backup_{timestamp}.png')
                os.rename(plot_png_path, backup_png)
        else:
            print(f"Error: Output files already exist. Use --resume to continue from previous run.")
            exit(1)

    # Initialize Plotter
    plotter = Plotter(
        log_file=scores_jsonl_path,
        update_interval=0.1,
        left_key="score",
        right_key=None,
        left_label="Caption Score",
        right_label=None,
        output_png=plot_png_path
    )

    # Start plotting in a background thread
    import threading
    plot_thread = threading.Thread(target=plotter.start_plotting)
    plot_thread.daemon = True
    plotter.running = True
    plot_thread.start()
    scores_by_epoch = []
    with open(scores_jsonl_path, "a") as f:
        for epoch, checkpoint_dir in tqdm(checkpoint_dirs, desc="Evaluating Checkpoints"):
            if epoch in completed_epochs:
                print(f"Skipping already evaluated checkpoint: {checkpoint_dir}")
                continue
                
            print(f"Evaluating checkpoint: {checkpoint_dir}")
            if using_unet_pipe:
                pipe = TextConditionalDDPMPipeline.from_pretrained(checkpoint_dir).to(device)
            else:
                pipe = FDMPipeline.from_pretrained(checkpoint_dir).to(device)


            avg_score, _, _, _ = calculate_caption_score_and_samples(
                device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, args.describe_absence, output=False, width=width, height=height
            )

            print(f"Checkpoint {checkpoint_dir} - Average caption adherence score: {avg_score:.4f}")
            result = {"epoch": epoch, "score": avg_score, "checkpoint_dir": checkpoint_dir}
            f.write(json.dumps(result) + "\n")
            f.flush()  # Ensure it's written immediately

            scores_by_epoch.append((epoch, avg_score, checkpoint_dir))

            # Update the plot after each checkpoint
            plotter.update_plot()

    plotter.stop_plotting()
    plot_thread.join(timeout=1)

    return scores_by_epoch

def calculate_caption_score_and_samples(device, pipe, dataloader, inference_steps, guidance_scale, random_seed, id_to_char, char_to_id, tile_descriptors, describe_absence, height, width, output=True):
    
    #With persistent_workers being set to true at creation, the dataset mode cannot be changed
    #without creating a new dataset, because all of the workers present are working with the old
    #version. This creates a new dataset to avoid this issue entierly
    original_mode = dataloader.dataset.mode
    if original_mode == "diff_text":
        dataset = dataloader.dataset
        dataset.mode = "text"
        dataloader = DataLoader(
            dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            persistent_workers=False
        )
        

    score_sum = 0.0
    total_count = 0
    all_samples = []
    all_prompts = []
    compare_all_scores = []
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():  # Disable gradient computation to save memory            
            if dataloader.dataset.negative_captions:
                # For negative captions, batch is (positive_captions, negative_captions)
                positive_captions, negative_captions = batch  # Unpack the batch directly
                param_values = {
                    "caption": list(positive_captions),
                    "negative_prompt": list(negative_captions),
                    "num_inference_steps": inference_steps,
                    "height": height,
                    "width": width,
                    "guidance_scale": guidance_scale,
                    "output_type": "tensor",
                    "batch_size": len(positive_captions)
                }
            elif isinstance(pipe, FDMPipeline):
                param_values = {
                    "caption": list(batch),
                    "batch_size": len(batch),
                }
            else:
                param_values = {
                    "caption": list(batch),
                    "num_inference_steps": inference_steps,
                    "height": height,
                    "width": width,
                    "guidance_scale": guidance_scale,
                    "output_type": "tensor",
                    "batch_size": len(batch)
                }

            generator = torch.Generator(device).manual_seed(int(random_seed))
            # Generate a batch of samples at once
            samples = pipe(generator=generator, **param_values).images  # (batch_size, ...)
            #print("samples.shape", samples.shape)
            for i in range(len(samples)):
                if dataloader.dataset.negative_captions:
                    caption = positive_captions[i]
                else:
                    caption = batch[i]
                    
                all_prompts.append(caption)

                sample = samples[i].unsqueeze(0)
                #print("sample.shape", sample.shape)
                sample_indices = convert_to_level_format(sample)
                #print("first sample_indices", sample_indices[0])
                scene = sample_indices[0].tolist()  # Always just one scene: (1,16,16)
                #quit()
                if height == common_settings.LR_HEIGHT:
                    actual_caption = lr_assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, describe_absence)
                elif height == common_settings.MARIO_HEIGHT:
                    actual_caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, describe_absence)

                if output: print(f"\t{caption}")

                compare_score = compare_captions(caption, actual_caption)

                if output: print(f"\tcompare_score: {compare_score}")
                compare_all_scores.append(compare_score)

                score_sum += compare_score
                total_count += 1

                all_samples.append(sample)  # Append the generated sample to the list
                del sample, sample_indices, scene, actual_caption  # Remove unused variables

        torch.cuda.empty_cache()  # Clear GPU VRAM cache

        if output: print(f"Batch {batch_idx+1}/{len(dataloader)}:")

    avg_score = score_sum / total_count
    # Concatenate all batches
    all_samples = torch.cat(all_samples, dim=0)[:total_count]

    dataloader.dataset.mode=original_mode

    return (avg_score, all_samples, all_prompts, compare_all_scores) 
    # Adding this return value broke code in MANY places. Cannot do this unless you make sure that all calls to this function expect 4 values
    # Found all references of this method and made them all return 4 values

if __name__ == "__main__":
    main()
