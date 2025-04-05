#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler, DDPMPipeline
from level_dataset import visualize_samples
import json
import random
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.colors as mcolors
from tokenizer import Tokenizer 
from models import TransformerModel
from text_diffusion_pipeline import TextConditionalDDPMPipeline

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
    
    # Visualization options
    parser.add_argument("--colormap", type=str, default="viridis", help="Matplotlib colormap for visualization")
    parser.add_argument("--figsize", type=int, nargs=2, default=[10, 10], help="Figure size for individual level plots")

    # Text conditional model    
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--mlm_model_file", type=str, default=os.path.join("mlm","mlm_transformer.pth"), help="Path to pre-trained text embedding model")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Text embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for text model")

    return parser.parse_args()

def convert_to_level_format(sample):
    """Convert model output to level indices"""
    sample_indices = torch.argmax(sample, dim=1).cpu().numpy()
    #print(sample_indices.shape)
    return sample_indices

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
    if True:
        tokenizer = Tokenizer()
        tokenizer.load(args.pkl)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_size = tokenizer.get_vocab_size()
        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim
        text_encoder = TransformerModel(vocab_size, embedding_dim, hidden_dim).to(device)
        text_encoder.load_state_dict(torch.load(args.mlm_model_file, map_location=device))
        text_encoder.eval()  # Set to evaluation mode
        pipeline = TextConditionalDDPMPipeline(
            unet=UNet2DConditionModel.from_pretrained(os.path.join(args.model_path, "unet")),
            scheduler=DDPMScheduler.from_pretrained(os.path.join(args.model_path, "scheduler")),
            text_encoder=text_encoder
        )
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

            # Convert shape if needed (DDPMPipeline might return different format)
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

    # Convert to list
    samples_list = [all_samples[i] for i in range(len(all_samples))]
    
    # Prepare a list to store all levels
    all_levels = []

    # Process and collect individual samples
    for _, sample in enumerate(samples_list):
        # Convert to indices
        sample_tensor = sample.unsqueeze(0) if sample.shape[0] == num_tiles else sample
        sample_indices = convert_to_level_format(sample_tensor)
        
        # Add level data to the list
        level_data = {
            "scene": sample_indices[0].tolist(), # Always just one scene: (1,16,16)
            "caption": "unknown" # TODO: extract this as I do in create_ascii_captions.py
        }
        all_levels.append(level_data)
    
    # Save all levels to a single JSON file if requested
    if args.save_as_json:
        json_path = os.path.join(args.output_dir, "all_levels.json")
        with open(json_path, 'w') as f:
            json.dump(all_levels, f, indent=2)
    
if __name__ == "__main__":
    args = parse_args()
    generate_levels(args)