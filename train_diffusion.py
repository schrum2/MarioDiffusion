import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup 
from tqdm.auto import tqdm
import pickle
import random
import numpy as np
from accelerate import Accelerator
import matplotlib
import matplotlib.pyplot as plt
from level_dataset import LevelDataset, visualize_samples
from tokenizer import Tokenizer 
import json
import threading
import time
from datetime import datetime
from loss_plotter import LossPlotter

# Create a custom pipeline for text-conditional generation
class TextConditionalDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, text_encoder=None):
        super().__init__(unet, scheduler)
        self.text_encoder = text_encoder
        
    def __call__(self, batch_size=1, generator=None, num_inference_steps=1000, 
                output_type="pil", captions=None, **kwargs):
        # Process text embeddings if captions are provided
        text_embeddings = None
        if captions is not None and self.text_encoder is not None:
            text_embeddings = self.text_encoder(captions)
        
        # Start from random noise
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("Must provide a generator for each sample if passing a list of generators")
            
        device = self.device
        sample_shape = (batch_size, self.unet.config.in_channels, 
                        self.unet.config.sample_size[0], self.unet.config.sample_size[1])
        
        if isinstance(generator, list):
            sample = torch.cat([
                torch.randn(1, *sample_shape[1:], generator=gen, device=device)
                for gen in generator
            ])
        else:
            sample = torch.randn(sample_shape, generator=generator, device=device)
            
        # Set number of inference steps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop
        for t in self.progress_bar(self.scheduler.timesteps):
            # Get model prediction
            model_input = torch.cat([sample] * 2) if text_embeddings is None else sample
            model_kwargs = {}
            if text_embeddings is not None:
                model_kwargs["encoder_hidden_states"] = text_embeddings
                
            # Predict noise residual
            noise_pred = self.unet(model_input, t, **model_kwargs).sample
            
            # Compute previous sample: x_{t-1} = scheduler(x_t, noise_pred)
            sample = self.scheduler.step(noise_pred, t, sample).prev_sample
            
        # Convert to output format
        if output_type == "pil":
            # Convert to PIL images
            # This would need to be adapted for one-hot level representation
            sample = (sample / 2 + 0.5).clamp(0, 1)
            sample = sample.cpu().permute(0, 2, 3, 1).numpy()
            
        elif output_type == "tensor":
            # Apply softmax to get probabilities for each tile type
            sample = F.softmax(sample, dim=1)
            
        return {"images": sample}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a text-conditional diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size") # TODO: Consider reducing to 16 to help generalization
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    
    # New text conditioning args
    parser.add_argument("--model_file", type=str, default=None, help="Path to pre-trained text embedding model")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Text embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for text model")
    parser.add_argument("--text_conditional", action="store_true", help="Enable text conditioning")
    parser.add_argument("--classifier_free_guidance_scale", type=float, default=7.5, 
                      help="Scale for classifier-free guidance during inference")
    
    # Model args
    parser.add_argument("--model_dim", type=int, default=128, help="Base dimension of UNet model")
    parser.add_argument("--dim_mults", nargs="+", type=int, default=[1, 2, 4], help="Dimension multipliers for UNet")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks per downsampling")
    parser.add_argument("--down_block_types", nargs="+", type=str, 
                       default=["DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"], 
                       help="Down block types for UNet")
    parser.add_argument("--up_block_types", nargs="+", type=str, 
                       default=["UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"], 
                       help="Up block types for UNet")
    
    # Training args
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr_warmup_percentage", type=float, default=0.05, help="Learning rate warmup portion") 
    parser.add_argument("--lr_scheduler_cycles", type=int, default=1, help="Number of cycles for the cosine learning rate scheduler")
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

    # Initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer.load(args.pkl)
    
    # Load text embedding model if text conditioning is enabled
    text_encoder = None
    if args.text_conditional and args.model_file:
        vocab_size = tokenizer.get_vocab_size()
        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim
        text_encoder = TransformerModel(vocab_size, embedding_dim, hidden_dim).to(device)
        text_encoder.load_state_dict(torch.load(args.model_file, map_location=device))
        text_encoder.eval()  # Set to evaluation mode
        print(f"Loaded text encoder from {args.model_file}")
    
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
    
    # Setup the UNet model - use conditional version if text conditioning is enabled
    if args.text_conditional:
        model = UNet2DConditionModel(
            sample_size=(16, 16),  # Fixed size for your level scenes
            in_channels=args.num_tiles,  # Number of tile types (for one-hot encoding)
            out_channels=args.num_tiles,
            layers_per_block=args.num_res_blocks,
            block_out_channels=[args.model_dim * mult for mult in args.dim_mults],
            down_block_types=args.down_block_types,
            up_block_types=args.up_block_types,
            cross_attention_dim=args.embedding_dim,  # Match the embedding dimension
        )
    else:
        model = UNet2DModel(
            sample_size=(16, 16),  # Fixed size for your level scenes
            in_channels=args.num_tiles,  # Number of tile types (for one-hot encoding)
            out_channels=args.num_tiles,
            layers_per_block=args.num_res_blocks,
            block_out_channels=[args.model_dim * mult for mult in args.dim_mults],
            down_block_types = [item.replace("CrossAttn", "") for item in args.down_block_types],
            up_block_types=[item.replace("CrossAttn", "") for item in args.up_block_types],
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
            # Process batch data
            if args.text_conditional:
                # Unpack scenes and captions
                scenes, captions = batch
                
                # Get text embeddings from the text encoder
                with torch.no_grad():
                    text_embeddings = text_encoder(captions)
                
                # For classifier-free guidance, we need to create a negative prompt embedding
                # We'll use the unconditional embedding
                uncond_tokens = torch.zeros_like(captions)
                with torch.no_grad():
                    uncond_embeddings = text_encoder(uncond_tokens)
            else:
                # For unconditional generation, we don't need captions
                if isinstance(batch, list):
                    scenes, _ = batch  # Ignore captions
                else:
                    scenes = batch

            # Add noise to the clean scenes
            noise = torch.randn_like(scenes)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (scenes.shape[0],), device=scenes.device).long()
            noisy_scenes = noise_scheduler.add_noise(scenes, noise, timesteps)
            
            with accelerator.accumulate(model):
                if args.text_conditional:
                    # Predict the noise with conditioning
                    noise_pred = model(noisy_scenes, timesteps, encoder_hidden_states=text_embeddings).sample
                else:
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
            
            global_step += 1
        
        # Generate and save sample levels every N epochs
        if epoch % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
            # Switch to eval mode
            model.eval()
            
            # Create the appropriate pipeline for generation
            if args.text_conditional:
                pipeline = TextConditionalDDPMPipeline(
                    unet=accelerator.unwrap_model(model), 
                    scheduler=noise_scheduler,
                    text_encoder=text_encoder
                )
                
                # Generate sample levels with conditioning
                sample_captions = [
                    "full floor. one enemy. a few question blocks. one platform at left middle. pipe at center bottom.",
                    "no floor. one enemy. a few coins. one coin line at right top. one platform at right top, one platform at right middle, one platform at left bottom.",
                    "floor with a few gaps. full ceiling. irregular block cluster at right top.",
                    "floor with one gap. full ceiling. two enemies. one ascending staircase. pipe at left bottom, pipe at left bottom."
                ]
                
                # Convert captions to token IDs using the tokenizer
                sample_caption_tokens = tokenizer.encode_batch(sample_captions)
                sample_caption_tokens = torch.tensor(sample_caption_tokens).to(accelerator.device)
                
                with torch.no_grad():
                    # Generate samples
                    samples = pipeline(
                        batch_size=4,
                        generator=torch.manual_seed(args.seed),
                        num_inference_steps=args.num_train_timesteps,
                        output_type="tensor",
                        captions=sample_caption_tokens
                    ).images
            else:
                # For unconditional generation
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

            # Convert one-hot samples to tile indices and visualize
            visualize_samples(samples, os.path.join(args.output_dir, f"samples_epoch_{epoch}"))
            
        # Save model every N epochs
        if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            # Save the model
            if args.text_conditional:
                pipeline = TextConditionalDDPMPipeline(
                    unet=accelerator.unwrap_model(model), 
                    scheduler=noise_scheduler,
                    text_encoder=text_encoder
                )
            else:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                
            pipeline.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{epoch}"))
    
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
    if args.text_conditional:
        pipeline = TextConditionalDDPMPipeline(
            unet=accelerator.unwrap_model(model), 
            scheduler=noise_scheduler,
            text_encoder=text_encoder
        )
    else:
        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
        
    pipeline.save_pretrained(args.output_dir)

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