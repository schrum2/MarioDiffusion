#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
from diffusers import DDPMPipeline
from level_dataset import visualize_samples, samples_to_scenes
import random
from text_diffusion_pipeline import TextConditionalDDPMPipeline
from create_ascii_captions import save_level_data

def parse_args():
    parser = argparse.ArgumentParser(description="Generate levels using a trained diffusion model")
    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of levels to generate")
    parser.add_argument("--output_dir", type=str, default="generated_levels", help="Directory to save generated levels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--inference_steps", type=int, default=500, help="Number of denoising steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--save_as_json", action="store_true", help="Save generated levels as JSON")
    parser.add_argument("--text_conditional", action="store_true", help="Enable text conditioning")

    # Used to generate captions when generating images
    parser.add_argument("--tileset", default='..\TheVGLC\Super Mario Bros\smb.json', help="Descriptions of individual tile types")
    parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")

    return parser.parse_args()

def generate_levels(args):
    """Generate level designs using a trained diffusion model"""
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the pipeline
    print(f"Loading model from {args.model_path}...")
    if args.text_conditional:
        pipeline = TextConditionalDDPMPipeline.from_pretrained(args.model_path)
    else:
        pipeline = DDPMPipeline.from_pretrained(args.model_path)
    pipeline.to(device)
    
    #print(pipeline)
    #print("---")
    #print(pipeline.unet)
    
    # Determine number of tiles from model
    num_tiles = pipeline.unet.config.in_channels
    print(f"Model configured for {num_tiles} tile types")
    
    # Generate in batches
    total_samples = args.num_samples
    num_batches = (total_samples + args.batch_size - 1) // args.batch_size
    
    all_samples = []
    
    for batch_idx in range(num_batches):
        # Calculate batch size for this iteration
        current_batch_size = min(args.batch_size, total_samples - batch_idx * args.batch_size)
        
        print(f"Generating batch {batch_idx+1}/{num_batches} ({current_batch_size} samples)...")
        
        # Generate samples
        with torch.no_grad():
            # Generate samples
            samples = pipeline(
                batch_size=current_batch_size,
                generator=torch.Generator(device).manual_seed(args.seed + batch_idx),
                num_inference_steps=args.inference_steps,
                output_type="tensor"
            ).images

            # Convert shape if needed (DDPMPipeline might return different format) (DO I EVEN NEED THIS?)
            if isinstance(samples, torch.Tensor):
                if len(samples.shape) == 4 and samples.shape[1] == 16:  # BHWC format
                    samples = samples.permute(0, 3, 1, 2)  # Convert (B, H, W, C) -> (B, C, H, W)
            elif isinstance(samples, np.ndarray):
                if len(samples.shape) == 4 and samples.shape[3] == num_tiles:  # BHWC format
                    samples = np.transpose(samples, (0, 3, 1, 2))  # Convert (B, H, W, C) -> (B, C, H, W)
                samples = torch.tensor(samples)

            #for i in range(16):
            #    for j in range(16):
            #        values = samples[0, :, i, j]  # Get channel values at (i, j)
            #        values = torch.tensor(values)
            #        max_idx = torch.argmax(values).item()
            #        print(f"({i},{j}): max idx={max_idx}, values={values.cpu().detach().numpy()}")

            all_samples.append(samples)
    
    # Concatenate all batches
    all_samples = torch.cat(all_samples, dim=0)[:total_samples]
    print(f"Generated {len(all_samples)} level samples")
    
    visualize_samples(all_samples, args.output_dir)

    if args.save_as_json:
        scenes = samples_to_scenes(all_samples)
        save_level_data(scenes, args.tileset, os.path.join(args.output_dir, "all_levels.json"), args.describe_locations, args.describe_absence)

if __name__ == "__main__":
    args = parse_args()
    generate_levels(args)