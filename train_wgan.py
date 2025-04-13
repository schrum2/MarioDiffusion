import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import random
import numpy as np
from level_dataset import LevelDataset, visualize_samples
import json
import threading
from datetime import datetime
from loss_plotter import LossPlotter
from tokenizer import Tokenizer 

def parse_args():
    parser = argparse.ArgumentParser(description="Train a text-conditional diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file (not used, but needed by LevelDataset)")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size") # TODO: Consider reducing to 16 to help generalization
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
            
    # Training args
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr_warmup_percentage", type=float, default=0.05, help="Learning rate warmup portion") 
    parser.add_argument("--lr_scheduler_cycles", type=float, default=0.5, help="Number of cycles for the cosine learning rate scheduler")
    parser.add_argument("--save_image_epochs", type=int, default=10, help="Save generated levels every N epochs")
    parser.add_argument("--save_model_epochs", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="level-gan-output", help="Output directory")
    
    # Diffusion scheduler args
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
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
    
    device = accelerator.device
        
    # Initialize tokenizer (LevelDataset constructor needs it, though captions are not used in training GAN)
    tokenizer = Tokenizer()
    tokenizer.load(args.pkl)

    # Initialize dataset
    dataset = LevelDataset(
        json_path=args.json,
        tokenizer=tokenizer,
        shuffle=True,
        mode="diffusion", # TODO: Not training a diffusion model, but LevelDataset needs this to return level scenes
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
    
    model = None # TODO: replace with GAN
        
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,  # Add weight decay to prevent overfitting
        betas=(0.9, 0.999)  # Default AdamW betas
    )
    
    # Setup learning rate scheduler
    total_training_steps = (len(dataloader) * args.num_epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_training_steps * args.lr_warmup_percentage)  

    print(f"Warmup period will be {warmup_steps} steps out of {total_training_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_cycles=args.lr_scheduler_cycles,
        num_warmup_steps=warmup_steps,  # Use calculated warmup steps
        num_training_steps=total_training_steps,
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
            # we don't need captions
            if isinstance(batch, list):
                scenes, _ = batch  # Ignore captions
            else: # Does this ever happen?
                scenes = batch

            # TODO: Model training
                      
            # Update progress bar
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
                        
            # Log to JSONL file
            log_metrics(epoch, loss.detach().item(), lr_scheduler.get_last_lr()[0], step=global_step)
            
            global_step += 1
        
        # Generate and save sample levels every N epochs
        if epoch % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
            # Switch to eval mode
            model.eval()
            
            # For GAN generation
            
            # TODO
            
            # First create samples somehow, then possibly manipulate like this, but may not be needed        
            samples = torch.tensor(samples).permute(0, 3, 1, 2)  # Convert (B, H, W, C) -> (B, C, H, W)

            # Convert one-hot samples to tile indices and visualize
            visualize_samples(samples, os.path.join(args.output_dir, f"samples_epoch_{epoch}"))
            
        # Save model every N epochs
        if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            # Save the model
            
            # TODO: Save somewhere like this: os.path.join(args.output_dir, f"checkpoint-{epoch}")
    
    # Clean up plotting resources
    if accelerator.is_local_main_process and plotter:
        # Better thread cleanup
        if plot_thread and plot_thread.is_alive():
            plotter.stop_plotting()
            plot_thread.join(timeout=5.0)
            if plot_thread.is_alive():
                print("Warning: Plot thread did not terminate properly")

    # Close progress bar and TensorBoard writer
    progress_bar.close()
    
    # Final model save
    # TODO: Final save to args.output_dir

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
