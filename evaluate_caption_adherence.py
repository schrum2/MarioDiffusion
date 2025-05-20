import argparse
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from level_dataset import LevelDataset, visualize_samples
import json
from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from level_dataset import visualize_samples, convert_to_level_format, samples_to_scenes
from create_ascii_captions import assign_caption, save_level_data, extract_tileset
from captions.caption_match import compare_captions
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate caption adherence for a pretrained text-conditional diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size") 
        
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--inference_steps", type=int, default=50, help="Number of denoising steps") # Large reduction from the 500 used during training
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--save_as_json", action="store_true", help="Save generated levels as JSON")

    # Used to generate captions when generating images
    parser.add_argument("--tileset", default='..\TheVGLC\Super Mario Bros\smb.json', help="Descriptions of individual tile types")
    #parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")

    # Output args
    parser.add_argument("--output_dir", type=str, default="text_to_level_results", help="Output directory if not comparing checkpoints (subdir of model directory)")


    parser.add_argument("--compare_checkpoints", action="store_true", default=False, help="Run comparison across all model checkpoints")

    return parser.parse_args()

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # Save within the model path directory
    if not args.compare_checkpoints:
        args.output_dir = os.path.join(args.model_path, args.output_dir)
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

    _, id_to_char, char_to_id, tile_descriptors = extract_tileset(args.tileset)
        
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # TODO: This won't work if training terminated early, but there are still valid checkpoints I want to evaluate
    pipe = TextConditionalDDPMPipeline.from_pretrained(args.model_path).to(device)

    # Initialize dataset
    dataset = LevelDataset(
        json_path=args.json,
        tokenizer=pipe.tokenizer,
        shuffle=False,
        mode="text",
        augment=False,
        num_tiles=args.num_tiles,
        block_embeddings=None
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
        scores_by_epoch = track_caption_adherence(args, device, dataloader, id_to_char, char_to_id, tile_descriptors)

        if args.save_as_json:
            # Save scores_by_epoch to a JSON file
            scores_json_path = os.path.join(args.model_path, f"{args.json.split('.')[0]}_scores_by_epoch.json")
            with open(scores_json_path, "w") as f:
                json.dump(scores_by_epoch, f, indent=4)
            print(f"Saved scores by epoch to {scores_json_path}")

        # Plot the scores
        import matplotlib.pyplot as plt

        epochs = [entry[0] for entry in scores_by_epoch]
        scores = [entry[1] for entry in scores_by_epoch]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, scores, marker="o", label="Caption Score")
        plt.xlabel("Epoch")
        plt.ylabel("Caption Score")
        plt.ylim(-1.0, 1.0)
        plt.title("Caption Adherence Score by Epoch")
        plt.grid(True)
        plt.legend()

        plot_path = os.path.join(args.model_path, f"{args.json.split('.')[0]}_caption_scores_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved caption scores plot to {plot_path}")
    else:
        # Just run on one model and get samples as well
        avg_score, all_samples = calculate_caption_score_and_samples(device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, args.describe_absence, output=False)

        print(f"Average caption adherence score: {avg_score:.4f}")
        print(f"Generated {len(all_samples)} level samples")
        
        visualize_samples(all_samples, args.output_dir)

        if args.save_as_json:
            scenes = samples_to_scenes(all_samples)
            save_level_data(scenes, args.tileset, os.path.join(args.output_dir, "all_levels.json"), False, args.describe_absence)


from util.plotter import Plotter  # Add this import at the top

def track_caption_adherence(args, device, dataloader, id_to_char, char_to_id, tile_descriptors):
    import json

    checkpoint_dirs = [
        (int(d.split("-")[-1]), os.path.join(args.model_path, d))
        for d in os.listdir(args.model_path)
        if os.path.isdir(os.path.join(args.model_path, d)) and d.startswith("checkpoint-")
    ]
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: x[0])
    if os.path.isdir(os.path.join(args.model_path, "unet")):
        checkpoint_dirs.append((checkpoint_dirs[-1][0] + 1, args.model_path))

    # Prepare output paths
    scores_jsonl_path = os.path.join(args.model_path, f"{args.json.split('.')[0]}_scores_by_epoch.jsonl")
    plot_png_path = os.path.join(args.model_path, f"{args.json.split('.')[0]}_caption_scores_plot.png")

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
            print(f"Evaluating checkpoint: {checkpoint_dir}")
            pipe = TextConditionalDDPMPipeline.from_pretrained(checkpoint_dir).to(device)

            avg_score, _ = calculate_caption_score_and_samples(
                device, pipe, dataloader, args.inference_steps, args.guidance_scale, args.seed, id_to_char, char_to_id, tile_descriptors, args.describe_absence, output=False
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

def calculate_caption_score_and_samples(device, pipe, dataloader, inference_steps, guidance_scale, random_seed, id_to_char, char_to_id, tile_descriptors, describe_absence, output=True):
    original_mode = dataloader.dataset.mode
    dataloader.dataset.mode = "text"  # Set mode to text for caption generation

    score_sum = 0.0
    total_count = 0
    all_samples = []
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():  # Disable gradient computation to save memory            
            if dataloader.dataset.negative_captions:
                # For negative captions, batch is (positive_captions, negative_captions)
                positive_captions, negative_captions = batch  # Unpack the batch directly
                param_values = {
                    "caption": list(positive_captions),
                    "negative_prompt": list(negative_captions),
                    "num_inference_steps": inference_steps,
                    "guidance_scale": guidance_scale,
                    "output_type": "tensor",
                    "batch_size": len(positive_captions)
                }
            else:
                param_values = {
                    "caption": list(batch),
                    "num_inference_steps": inference_steps,
                    "guidance_scale": guidance_scale,
                    "output_type": "tensor",
                    "batch_size": len(batch)
                }

            generator = torch.Generator(device).manual_seed(int(random_seed))
            # Generate a batch of samples at once
            samples = pipe(generator=generator, **param_values).images  # (batch_size, ...)

            for i in range(len(samples)):
                if dataloader.dataset.negative_captions:
                    caption = positive_captions[i]
                else:
                    caption = batch[i]

                sample = samples[i].unsqueeze(0)
                sample_indices = convert_to_level_format(sample)
                scene = sample_indices[0].tolist()  # Always just one scene: (1,16,16)
                actual_caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, describe_absence)

                if output: print(f"\t{caption}")

                compare_score = compare_captions(caption, actual_caption)

                score_sum += compare_score
                total_count += 1

                all_samples.append(sample)  # Append the generated sample to the list
                del sample, sample_indices, scene, actual_caption  # Remove unused variables

        torch.cuda.empty_cache()  # Clear GPU VRAM cache

        if output: print(f"Batch {batch_idx+1}/{len(dataloader)}:")

    avg_score = score_sum / total_count
    # Concatenate all batches
    all_samples = torch.cat(all_samples, dim=0)[:total_count]

    # Reset mode to original
    dataloader.dataset.mode = original_mode

    return (avg_score, all_samples)

if __name__ == "__main__":
    main()
