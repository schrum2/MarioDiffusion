import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import pickle
import json
import random
import numpy as np
from accelerate import Accelerator
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from level_dataset import LevelDataset
from tokenizer import Tokenizer 

def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    
    # Model args
    parser.add_argument("--model_dim", type=int, default=128, help="Base dimension of UNet model")
    parser.add_argument("--dim_mults", nargs="+", type=int, default=[1, 2, 4], help="Dimension multipliers for UNet")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks per downsampling")
    parser.add_argument("--down_block_types", nargs="+", type=str, 
                       default=["DownBlock2D", "AttnDownBlock2D", "DownBlock2D"], 
                       help="Down block types for UNet")
    parser.add_argument("--up_block_types", nargs="+", type=str, 
                       default=["UpBlock2D", "AttnUpBlock2D", "UpBlock2D"], 
                       help="Up block types for UNet")
    parser.add_argument("--add_attention", action="store_true", help="Add attention layers to the model")
    
    # Training args
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Learning rate warmup steps")
    parser.add_argument("--save_image_epochs", type=int, default=10, help="Save generated levels every N epochs")
    parser.add_argument("--save_model_epochs", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="level-diffusion-output", help="Output directory")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory for TensorBoard")
    
    # Diffusion scheduler args
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--beta_schedule", type=str, default="linear", help="Beta schedule type")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Beta schedule start value")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta schedule end value")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.logging_dir,
    )
    
    # Initialize the TensorBoard writer
    writer = SummaryWriter(log_dir=args.logging_dir)
    
    tokenizer = Tokenizer()
    tokenizer.load('SMB1_Tokenizer.pkl')
    
    # Initialize dataset
    dataset = LevelDataset(
        json_path=args.json,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        mode="diffusion",
        augment=args.augment,
        num_tiles=args.num_tiles
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    # Setup the UNet model
    model = UNet2DModel(
        sample_size=(16, 16),  # Fixed size for your level scenes
        in_channels=args.num_tiles,  # Number of tile types (for one-hot encoding)
        out_channels=args.num_tiles,
        layers_per_block=args.num_res_blocks,
        block_out_channels=[args.model_dim * mult for mult in args.dim_mults],
        down_block_types=args.down_block_types,
        up_block_types=args.up_block_types,
    )
    
    # Setup the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.num_epochs) // args.gradient_accumulation_steps,
    )
    
    # Prepare for training with accelerator
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(total=args.num_epochs * len(dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(args.num_epochs):
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            # We're ignoring captions for unconditional generation
            if isinstance(batch, tuple):
                scenes, _ = batch
            else:
                scenes = batch
            
            # Add noise to the clean scenes
            noise = torch.randn_like(scenes)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (scenes.shape[0],), device=scenes.device).long()
            noisy_scenes = noise_scheduler.add_noise(scenes, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise
                noise_pred = model(noisy_scenes, timesteps).sample
                
                # Compute loss
                loss = F.mse_loss(noise_pred, noise)
                
                # Backpropagation
                accelerator.backward(loss)
                
                # Update the model parameters
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1
            
            # Log to TensorBoard
            accelerator.log(logs, step=global_step)
        
        # Generate and save sample levels every N epochs
        if epoch % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
            # Switch to eval mode
            model.eval()
            
            # Create a pipeline for generation
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            
            # Generate sample levels
            with torch.no_grad():
                # Sample random noise
                sample = torch.randn(
                    4, args.num_tiles, 16, 16,
                    generator=torch.manual_seed(args.seed),
                    device=accelerator.device
                )
                
                # Generate samples from noise
                samples = pipeline(
                    batch_size=4,
                    generator=torch.manual_seed(args.seed),
                    num_inference_steps=args.num_train_timesteps,
                    output_type="tensor",
                ).images
            
            # Convert one-hot samples to tile indices
            samples_indices = visualize_samples(samples, dataset, args.output_dir, epoch)
            
            # Log samples to TensorBoard
            for i, sample_level in enumerate(samples_indices):
                fig = plt.figure(figsize=(5, 5))
                plt.imshow(sample_level, cmap='viridis')
                plt.colorbar(label='Tile Type')
                plt.tight_layout()
                writer.add_figure(f"generated_level_{i}", fig, global_step=epoch)
        
        # Save model every N epochs
        if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            # Save the model
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            pipeline.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{epoch}"))
    
    # Close progress bar and TensorBoard writer
    progress_bar.close()
    writer.close()
    
    # Final model save
    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
    pipeline.save_pretrained(args.output_dir)

def visualize_samples(samples, dataset, output_dir, epoch):
    """
    Visualize generated samples and save as images.
    
    Args:
        samples: One-hot encoded samples from the diffusion model
        dataset: LevelDataset instance for decoding
        output_dir: Directory to save visualizations
        epoch: Current epoch number
    
    Returns:
        List of tile index maps for the samples
    """
    # Create directory for this epoch's samples
    samples_dir = os.path.join(output_dir, f"samples_epoch_{epoch}")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Convert from one-hot to tile indices
    sample_indices = []
    plt.figure(figsize=(16, 4))
    
    for i, sample in enumerate(samples):
        # Convert one-hot back to indices (get most likely tile for each position)
        # [num_tiles, height, width] -> [height, width]
        sample_index = torch.argmax(sample, dim=0).cpu().numpy()
        sample_indices.append(sample_index)
        
        # Plot and save
        plt.subplot(1, 4, i + 1)
        plt.imshow(sample_index, cmap='viridis')
        plt.colorbar(label='Tile Type')
        plt.title(f"Sample {i+1}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(samples_dir, "samples_grid.png"))
    plt.close()
    
    # Save individual samples
    for i, sample_index in enumerate(sample_indices):
        plt.figure(figsize=(8, 8))
        plt.imshow(sample_index, cmap='viridis')
        plt.colorbar(label='Tile Type')
        plt.title(f"Sample {i+1}")
        plt.savefig(os.path.join(samples_dir, f"sample_{i}.png"))
        plt.close()
    
    return sample_indices

def generate_levels(model_path, num_samples=10, output_dir="generated_levels", seed=42):
    """
    Generate new level designs using a trained diffusion model.
    
    Args:
        model_path: Path to the saved diffusion model
        num_samples: Number of levels to generate
        output_dir: Directory to save generated levels
        seed: Random seed for reproducibility
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Load the pipeline
    pipeline = DDPMPipeline.from_pretrained(model_path)
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples
    print(f"Generating {num_samples} level samples...")
    samples = pipeline(
        batch_size=num_samples,
        generator=torch.manual_seed(seed),
        num_inference_steps=1000,
        output_type="tensor"
    ).images
    
    # Visualize and save samples
    plt.figure(figsize=(20, 4 * ((num_samples + 4) // 5)))
    
    for i, sample in enumerate(samples):
        # Convert one-hot back to indices
        sample_index = torch.argmax(sample, dim=0).cpu().numpy()
        
        # Plot
        plt.subplot((num_samples + 4) // 5, 5, i + 1)
        plt.imshow(sample_index, cmap='viridis')
        plt.colorbar(label='Tile Type')
        plt.title(f"Level {i+1}")
        
        # Save individual sample
        plt.figure(figsize=(8, 8))
        plt.imshow(sample_index, cmap='viridis')
        plt.colorbar(label='Tile Type')
        plt.title(f"Level {i+1}")
        plt.savefig(os.path.join(output_dir, f"level_{i}.png"))
        plt.close()
    
    # Save grid of all samples
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "levels_grid.png"))
    plt.close()
    
    print(f"Generated {num_samples} levels saved to {output_dir}")

if __name__ == "__main__":
    main()