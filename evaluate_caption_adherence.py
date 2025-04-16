import argparse
import os
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from level_dataset import LevelDataset, visualize_samples
import json
from text_diffusion_pipeline import TextConditionalDDPMPipeline
from level_dataset import visualize_samples, convert_to_level_format, samples_to_scenes
import json
import random
from text_diffusion_pipeline import TextConditionalDDPMPipeline
from create_ascii_captions import assign_caption, get_tile_descriptors, save_level_data
from caption_match import compare_captions

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
    parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")

    # Output args
    parser.add_argument("--output_dir", type=str, default="text_to_level_results", help="Output directory")


    parser.add_argument("--compare_checkpoints", action="store_true", default=False, help="Run comparison across all model checkpoints")

    return parser.parse_args()

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.tileset, "r") as f:
        tileset = json.load(f)
        tile_chars = sorted(tileset['tiles'].keys())
        id_to_char = {idx: char for idx, char in enumerate(tile_chars)}
        char_to_id = {char: idx for idx, char in enumerate(tile_chars)}
        tile_descriptors = get_tile_descriptors(tileset)    
        
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pipe = TextConditionalDDPMPipeline.from_pretrained(args.model_path).to(device)

    # Initialize dataset
    dataset = LevelDataset(
        json_path=args.json,
        tokenizer=pipe.text_encoder.tokenizer,
        shuffle=False,
        mode="diffusion",
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
        scores_by_epoch = track_caption_adherence(args, device, dataloader, id_to_char, char_to_id, tile_descriptors)

        # Save scores_by_epoch to a JSON file
        scores_json_path = os.path.join(args.output_dir, "scores_by_epoch.json")
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

        plot_path = os.path.join(args.output_dir, "caption_scores_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved caption scores plot to {plot_path}")
    else:
        # Just run on one model and get samples as well
        avg_score, all_samples = calculate_caption_score_and_samples(args, device, pipe, dataloader, id_to_char, char_to_id, tile_descriptors)

        print(f"Average caption adherence score: {avg_score:.4f}")
        print(f"Generated {len(all_samples)} level samples")
    
        visualize_samples(all_samples, args.output_dir)

        if args.save_as_json:
            scenes = samples_to_scenes(all_samples)
            save_level_data(scenes, args.tileset, os.path.join(args.output_dir, "all_levels.json"), args.describe_locations, args.describe_absence)


def track_caption_adherence(args, device, dataloader, id_to_char, char_to_id, tile_descriptors):
    checkpoint_dirs = [
        (int(d.split("-")[-1]), os.path.join(args.model_path, d))
        for d in os.listdir(args.model_path)
        if os.path.isdir(os.path.join(args.model_path, d)) and d.startswith("checkpoint-")
    ]
    
    # Sort directories by epoch number
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: x[0])
    # Final model is saved in the output directory itself rather than a subdirectory
    if os.path.isdir(os.path.join(args.model_path, "unet")): # Make sure final successful save occurred
        checkpoint_dirs.append(checkpoint_dirs[-1][0] + 1 , args.model_path)

    scores_by_epoch = []
    for epoch, checkpoint_dir in checkpoint_dirs:
        print(f"Evaluating checkpoint: {checkpoint_dir}")
        pipe = TextConditionalDDPMPipeline.from_pretrained(checkpoint_dir).to(device)
        
        avg_score, _ = calculate_caption_score_and_samples(
            args, device, pipe, dataloader, id_to_char, char_to_id, tile_descriptors
        )
        
        print(f"Checkpoint {checkpoint_dir} - Average caption adherence score: {avg_score:.4f}")
        scores_by_epoch.append((epoch, avg_score, checkpoint_dir))

    return scores_by_epoch

def calculate_caption_score_and_samples(args, device, pipe, dataloader, id_to_char, char_to_id, tile_descriptors):
    score_sum = 0.0
    total_count = 0
    all_samples = []
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():  # Disable gradient computation to save memory
            # Unpack scenes and captions
            _, sample_caption_tokens = batch

            param_values = {
                "captions" : sample_caption_tokens.to(device),
                "num_inference_steps": args.inference_steps,
                "guidance_scale": args.guidance_scale,
                #"width": 16, # Might consider changing this later
                "output_type" : "tensor",
                "batch_size" : len(sample_caption_tokens)
            }
            generator = torch.Generator(device).manual_seed(int(args.seed))
            
            samples = pipe(generator=generator, **param_values).images

            # Convert shape if needed (DO I EVEN NEED THIS?)
            if isinstance(samples, torch.Tensor):
                if len(samples.shape) == 4 and samples.shape[1] == 16:  # BHWC format
                    samples = samples.permute(0, 3, 1, 2)  # Convert (B, H, W, C) -> (B, C, H, W)
            elif isinstance(samples, np.ndarray):
                if len(samples.shape) == 4 and samples.shape[3] == args.num_tiles:  # BHWC format
                    samples = np.transpose(samples, (0, 3, 1, 2))  # Convert (B, H, W, C) -> (B, C, H, W)
                samples = torch.tensor(samples)        
            
            # Iterate over captions and corresponding generated images
            for caption_tokens, image in zip(sample_caption_tokens, samples):
                sample_tensor = image.unsqueeze(0)
                sample_indices = convert_to_level_format(sample_tensor)
                scene = sample_indices[0].tolist()  # Always just one scene: (1,16,16)
                actual_caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, args.describe_locations, args.describe_absence)

                caption = pipe.text_encoder.tokenizer.decode(caption_tokens.tolist())
                caption = caption.replace("[PAD]", "").replace(" .", ".").strip()
                print(f"\t{caption}")

                compare_score = compare_captions(caption, actual_caption)

                score_sum += compare_score
                total_count += 1

            all_samples.append(samples)  # Append the generated samples to the list
            del samples, sample_tensor, sample_indices, scene, actual_caption  # Remove unused variables

        torch.cuda.empty_cache()  # Clear GPU VRAM cache

        print(f"Batch {batch_idx+1}/{len(dataloader)}:")

    avg_score = score_sum / total_count
    # Concatenate all batches
    all_samples = torch.cat(all_samples, dim=0)[:total_count]

    return (avg_score, all_samples)

if __name__ == "__main__":
    main()
