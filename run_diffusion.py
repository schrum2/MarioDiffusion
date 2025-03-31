#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline
from tokenizer import Tokenizer
import json
import random
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.colors as mcolors

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
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer file for visualization")
    
    # Visualization options
    parser.add_argument("--colormap", type=str, default="viridis", help="Matplotlib colormap for visualization")
    parser.add_argument("--figsize", type=int, nargs=2, default=[10, 10], help="Figure size for individual level plots")
    
    return parser.parse_args()

def setup_visualizer(tokenizer_path=None):
    """Set up visualization utilities with optional tokenizer for better rendering"""
    if tokenizer_path and os.path.exists(tokenizer_path):
        try:
            tokenizer = Tokenizer()
            tokenizer.load(tokenizer_path)
            return tokenizer
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
    return None

def create_custom_colormap(num_tiles):
    """Create a custom colormap that makes different tiles visually distinct"""
    # Generate evenly spaced colors across HSV space
    colors = plt.cm.get_cmap('hsv', num_tiles)(np.linspace(0, 1, num_tiles))
    
    # Create a special color for empty space (typically index 0)
    colors[0] = [0.95, 0.95, 0.95, 1.0]  # Light gray for empty space
    
    return mcolors.ListedColormap(colors)

def visualize_level(level_indices, tokenizer=None, cmap='viridis', title="Generated Level", figsize=(10, 10), save_path=None):
    """Visualize a single level with optional tokenizer labels"""
    plt.figure(figsize=figsize)
    
    num_tiles = level_indices.max() + 1
    custom_cmap = create_custom_colormap(num_tiles)
    
    # Plot the level with colormap
    img = plt.imshow(level_indices, cmap=custom_cmap, interpolation='nearest')
    plt.colorbar(img, label='Tile Type')
    plt.title(title)
    
    # Add grid lines for visibility
    plt.grid(which='both', color='lightgrey', linestyle='-', linewidth=0.5)
    
    # If tokenizer is available, add tile names as text annotations
    if tokenizer:
        height, width = level_indices.shape
        for y in range(height):
            for x in range(width):
                tile_idx = level_indices[y, x]
                if tile_idx > 0:  # Skip annotating empty tiles
                    try:
                        tile_name = tokenizer.id_to_token.get(int(tile_idx), str(tile_idx))
                        # Abbreviate long tile names
                        if len(tile_name) > 10:
                            tile_name = tile_name[:8] + ".."
                        
                        # Choose text color based on background brightness
                        color_val = custom_cmap(tile_idx / (num_tiles - 1))
                        brightness = 0.299 * color_val[0] + 0.587 * color_val[1] + 0.114 * color_val[2]
                        text_color = 'black' if brightness > 0.5 else 'white'
                        
                        plt.text(x, y, tile_name, ha='center', va='center', 
                                 color=text_color, fontsize=8)
                    except Exception as e:
                        print(f"Warning: Could not annotate tile at ({x}, {y}): {e}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def convert_to_level_format(sample):
    """Convert model output to level indices"""
    if isinstance(sample, np.ndarray):
        # Handle numpy arrays
        if len(sample.shape) == 4:  # BCHW format
            sample_indices = np.argmax(sample, axis=1)
        elif len(sample.shape) == 3 and sample.shape[0] > 1:  # CHW format (single sample)
            sample_indices = np.argmax(sample, axis=0)
        else:
            sample_indices = sample
        return sample_indices
    else:
        # Handle torch tensors
        if len(sample.shape) == 4:  # BCHW format
            sample_indices = torch.argmax(sample, dim=1).cpu().numpy()
        elif len(sample.shape) == 3 and sample.shape[0] > 1:  # CHW format (single sample)
            sample_indices = torch.argmax(sample, dim=0).cpu().numpy()
        else:
            sample_indices = sample.cpu().numpy()
        return sample_indices

def save_level_as_json(level_indices, save_path, tokenizer=None):
    """Save generated level in a JSON format"""
    level_data = {
        "width": level_indices.shape[1],
        "height": level_indices.shape[0],
        "tiles": level_indices.tolist()
    }
    
    # Add tile names if tokenizer is available
    if tokenizer:
        tile_names = {}
        for idx in np.unique(level_indices):
            tile_names[int(idx)] = tokenizer.id_to_token.get(int(idx), f"unknown_{idx}")
        level_data["tile_names"] = tile_names
    
    with open(save_path, 'w') as f:
        json.dump(level_data, f, indent=2)

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
    
    # Load the tokenizer if available
    tokenizer = setup_visualizer(args.pkl)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the pipeline
    print(f"Loading model from {args.model_path}...")
    pipeline = DDPMPipeline.from_pretrained(args.model_path)
    pipeline.to(device)
    
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
            
            # Convert shape if needed (DDPMPipeline might return different format)
            if isinstance(samples, torch.Tensor):
                if len(samples.shape) == 4 and samples.shape[1] == 16:  # BHWC format
                    samples = samples.permute(0, 3, 1, 2)  # Convert (B, H, W, C) -> (B, C, H, W)
            elif isinstance(samples, np.ndarray):
                if len(samples.shape) == 4 and samples.shape[3] == num_tiles:  # BHWC format
                    samples = np.transpose(samples, (0, 3, 1, 2))  # Convert (B, H, W, C) -> (B, C, H, W)
                samples = torch.tensor(samples)
            
            all_samples.append(samples)
    
    # Concatenate all batches
    try:
        all_samples = torch.cat(all_samples, dim=0)[:total_samples]
    except RuntimeError as e:
        print(f"Error concatenating samples: {e}")
        # Alternative approach if cat fails
        all_samples = [sample for batch in all_samples for sample in batch][:total_samples]
    
    print(f"Generated {len(all_samples)} level samples")
    
    # Convert to list of numpy arrays if it's not already
    if isinstance(all_samples, torch.Tensor):
        samples_list = [all_samples[i] for i in range(len(all_samples))]
    else:
        samples_list = all_samples
    
    # Visualize and save individual samples
    for i, sample in enumerate(samples_list):
        # Convert to indices
        if isinstance(sample, torch.Tensor) and len(sample.shape) == 3:
            sample_tensor = sample.unsqueeze(0) if sample.shape[0] == num_tiles else sample
            sample_indices = convert_to_level_format(sample_tensor)
            if sample_indices.shape[0] == 1:
                sample_indices = sample_indices[0]
        else:
            # Try to handle any other format
            sample_indices = convert_to_level_format(sample)
            if len(sample_indices.shape) > 2:
                sample_indices = sample_indices[0]
        
        # Save visualization
        img_path = os.path.join(args.output_dir, f"level_{i}.png")
        visualize_level(
            sample_indices,
            tokenizer=tokenizer, 
            cmap=args.colormap,
            title=f"Generated Level {i+1}",
            figsize=tuple(args.figsize),
            save_path=img_path
        )
        
        # Save as JSON if requested
        if args.save_as_json:
            json_path = os.path.join(args.output_dir, f"level_{i}.json")
            save_level_as_json(sample_indices, json_path, tokenizer)
    
    # Create a grid visualization of all levels
    rows = int(np.ceil(np.sqrt(len(samples_list))))
    cols = int(np.ceil(len(samples_list) / rows))
    
    plt.figure(figsize=(cols * 3, rows * 3))
    
    for i, sample in enumerate(samples_list):
        # Convert to indices again
        if isinstance(sample, torch.Tensor) and len(sample.shape) == 3:
            sample_tensor = sample.unsqueeze(0) if sample.shape[0] == num_tiles else sample
            sample_indices = convert_to_level_format(sample_tensor)
            if sample_indices.shape[0] == 1:
                sample_indices = sample_indices[0]
        else:
            sample_indices = convert_to_level_format(sample)
            if len(sample_indices.shape) > 2:
                sample_indices = sample_indices[0]
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(sample_indices, cmap=args.colormap)
        plt.title(f"Level {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "levels_grid.png"), dpi=150)
    plt.close()
    
    print(f"All generated levels saved to {args.output_dir}")
    print(f"Grid visualization saved as {os.path.join(args.output_dir, 'levels_grid.png')}")

if __name__ == "__main__":
    args = parse_args()
    generate_levels(args)