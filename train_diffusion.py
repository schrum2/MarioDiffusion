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
import random
import numpy as np
from accelerate import Accelerator
import matplotlib
import matplotlib.pyplot as plt
from level_dataset import LevelDataset
from tokenizer import Tokenizer 
import json
import threading
import time
from datetime import datetime

# Create a loss plotter class
class LossPlotter:
    def __init__(self, log_file, update_interval=1.0):
        self.log_file = log_file
        self.update_interval = update_interval
        self.running = True
        
        # Use non-interactive backend
        matplotlib.use('Agg')
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.epochs = []
        self.losses = []
        self.lr_values = []
        
    def update_plot(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    data = [json.loads(line) for line in f if line.strip()]
                    
                if not data:
                    return
                    
                self.epochs = [entry.get('step', 0) for entry in data]
                self.losses = [entry.get('loss', 0) for entry in data]
                self.lr_values = [entry.get('lr', 0) for entry in data]
                
                # Clear the axes and redraw
                self.ax.clear()
                # Plot loss
                self.ax.plot(self.epochs, self.losses, 'b-', label='Training Loss')
                self.ax.set_xlabel('Step')
                self.ax.set_ylabel('Loss', color='b')
                self.ax.tick_params(axis='y', labelcolor='b')
                
                # Add learning rate on secondary y-axis
                if any(self.lr_values):
                    ax2 = self.ax.twinx()
                    ax2.plot(self.epochs, self.lr_values, 'r-', label='Learning Rate')
                    ax2.set_ylabel('Learning Rate', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    ax2.legend(loc='upper right')
                
                # Add a title and legend
                self.ax.set_title('Training Progress')
                self.ax.legend(loc='upper left')
                
                # Adjust layout
                self.fig.tight_layout()
                
                # Save the current plot to disk
                self.fig.savefig(os.path.join(os.path.dirname(self.log_file), 'training_progress.png'))
            
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing log file: {e}")
    
    def start_plotting(self):
        """Method for non-interactive plotting to run in thread"""
        print("Starting non-interactive plotting in background")
        while self.running:
            self.update_plot()
            time.sleep(self.update_interval)
    
    def stop_plotting(self):
        self.running = False
        plt.close(self.fig)
        print("Plotting stopped")

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
    parser.add_argument("--lr_warmup_steps", type=int, default=10, help="Learning rate warmup steps")
    parser.add_argument("--save_image_epochs", type=int, default=10, help="Save generated levels every N epochs")
    parser.add_argument("--save_model_epochs", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="level-diffusion-output", help="Output directory")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory for TensorBoard")
    
    # Diffusion scheduler args
    parser.add_argument("--num_train_timesteps", type=int, default=500, help="Number of diffusion timesteps")
    parser.add_argument("--beta_schedule", type=str, default="linear", help="Beta schedule type")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Beta schedule start value")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta schedule end value")
    
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file with training parameters.")

    return parser.parse_args()

def main():
    args = parse_args()

    # Check if config file is provided before training loop begins
    if hasattr(args, 'config') and args.config:
        config = load_config_from_json(args.config)
        args = update_args_from_config(args, config)
        print("Training will use parameters from the config file.")
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

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
    
    # Get formatted timestamp for filenames
    formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')

    # Create log files
    log_file = os.path.join(args.output_dir, f"training_log_{formatted_date}.jsonl")
    config_file = os.path.join(args.output_dir, f"hyperparams_{formatted_date}.json")

    # Save hyperparameters to JSON file
    if accelerator.is_local_main_process:
        hyperparams = vars(args)
        with open(config_file, "w") as f:
            json.dump(hyperparams, f, indent=4)
        print(f"Saved configuration to: {config_file}")
  
    # Add function to log metrics
    def log_metrics(epoch, loss, lr, step=None):
        if accelerator.is_local_main_process:
            log_entry = {
                "epoch": epoch,
                "loss": loss,
                "lr": lr,
                "step": step if step is not None else epoch * len(dataloader),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    # Initialize plotter if we're on the main process
    plotter = None
    plot_thread = None
    if accelerator.is_local_main_process:
        plotter = LossPlotter(log_file, update_interval=5.0)  # Update every 5 seconds
        plot_thread = threading.Thread(target=plotter.start_plotting)
        plot_thread.daemon = True
        plot_thread.start()
        print(f"Loss plotting enabled. Progress will be saved to {os.path.join(args.output_dir, 'training_progress.png')}")

    for epoch in range(args.num_epochs):
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            # We're ignoring captions for unconditional generation
            if isinstance(batch, list):
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
                        
            # Log to JSONL file
            log_metrics(epoch, loss.detach().item(), lr_scheduler.get_last_lr()[0], step=global_step)
            print(logs)

            global_step += 1
        
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
                    generator=torch.Generator(accelerator.device).manual_seed(args.seed),
                    device=accelerator.device
                )

                # Generate samples from noise
                samples = pipeline(
                    batch_size=4,
                    generator=torch.manual_seed(args.seed),
                    num_inference_steps=args.num_train_timesteps,
                    output_type="tensor",
                ).images
            
                samples = torch.tensor(samples).permute(0, 3, 1, 2)  # Convert (B, H, W, C) -> (B, C, H, W)

            # Convert one-hot samples to tile indices
            samples_indices = dataset.visualize_samples(samples, args.output_dir, f"samples_epoch_{epoch}")
            
        # Save model every N epochs
        if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            # Save the model
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            pipeline.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{epoch}"))
    
    # Clean up plotting resources
    if accelerator.is_local_main_process and plotter:
        plotter.stop_plotting()
        if plot_thread and plot_thread.is_alive():
            plot_thread.join(timeout=1.0)

    # Close progress bar and TensorBoard writer
    progress_bar.close()
    
    # Final model save
    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
    pipeline.save_pretrained(args.output_dir)

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

# Add function to load config from JSON
def load_config_from_json(config_path):
    """Load hyperparameters from a JSON config file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Configuration loaded from {config_path}")
            
            # Print the loaded config for verification
            print("Loaded hyperparameters:")
            for key, value in config.items():
                print(f"  {key}: {value}")
                
            return config
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading config file: {e}")
        raise e

def update_args_from_config(args, config):
    """Update argparse namespace with values from config."""
    # Convert config dict to argparse namespace
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args

if __name__ == "__main__":
    main()