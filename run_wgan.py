import argparse
import os
import torch
import numpy as np
from wgan_model import WGAN_Generator
from tokenizer import Tokenizer
from level_dataset import visualize_samples
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from a trained WGAN generator")
    
    # Model loading args
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved generator model")
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    
    # Generation args
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--nz", type=int, default=32, help="Size of the latent z vector")
    parser.add_argument("--ngf", type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument("--n_extra_layers", type=int, default=0, help="Number of extra layers in generator")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="generated_samples", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer (needed for visualization)
    tokenizer = Tokenizer()
    tokenizer.load(args.pkl)
    
    # Set input image size (16x16 for your level data)
    isize = 16
    
    # Initialize generator
    netG = WGAN_Generator(isize, args.nz, args.num_tiles, args.ngf, n_extra_layers=args.n_extra_layers)
    
    # Load trained model
    try:
        netG.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded generator model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Move model to device and set to evaluation mode
    netG = netG.to(device)
    netG.eval()
    
    # Calculate number of batches needed
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    samples_generated = 0
    
    print(f"Generating {args.num_samples} samples...")
    
    # Generate samples in batches
    for batch_idx in range(num_batches):
        # Calculate batch size for the current batch
        current_batch_size = min(args.batch_size, args.num_samples - samples_generated)
        
        # Generate random noise
        noise = torch.randn(current_batch_size, args.nz, 1, 1, device=device)
        
        # Generate samples
        with torch.no_grad():
            fake_samples = netG(noise)
        
        # Convert samples to the right format for visualization
        samples_cpu = fake_samples.detach().cpu()
        
        # Visualize and save samples
        output_path = os.path.join(args.output_dir, f"samples_batch_{batch_idx}")
        visualize_samples(samples_cpu, output_path, save_individual=True)
        
        # Update counter
        samples_generated += current_batch_size
        print(f"Generated {samples_generated}/{args.num_samples} samples")
    
    print(f"Sample generation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
