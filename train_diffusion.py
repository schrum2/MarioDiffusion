import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup 
from tqdm.auto import tqdm
import random
import numpy as np
from accelerate import Accelerator
from level_dataset import LevelDataset, visualize_samples
from tokenizer import Tokenizer 
import json
import threading
from datetime import datetime
from util.loss_plotter import LossPlotter
from models.text_model import TransformerModel
from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from models.latent_diffusion_pipeline import UnconditionalDDPMPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Train a text-conditional diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size") # TODO: Consider reducing to 16 to help generalization
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument('--split', action='store_true', help='Enable train/val/test split')
    parser.add_argument('--train_pct', type=float, default=0.7, help='Train split percentage (default 0.7)')
    parser.add_argument('--val_pct', type=float, default=0.1, help='Validation split percentage (default 0.1)')
    parser.add_argument('--test_pct', type=float, default=0.2, help='Test split percentage (default 0.2)')
    
    # New text conditioning args
    parser.add_argument("--mlm_model_dir", type=str, default="mlm", help="Path to pre-trained text embedding model")
    parser.add_argument("--text_conditional", action="store_true", help="Enable text conditioning")
    parser.add_argument("--negative_prompt_training", action="store_true", help="Enable training with negative prompts")
    
    # Model args
    parser.add_argument("--model_dim", type=int, default=128, help="Base dimension of UNet model")
    parser.add_argument("--dim_mults", nargs="+", type=int, default=[1, 2, 4], help="Dimension multipliers for UNet")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks per downsampling")
    parser.add_argument("--down_block_types", nargs="+", type=str, 
                       default=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"], 
                       help="Down block types for UNet")
    parser.add_argument("--up_block_types", nargs="+", type=str, 
                       default=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"], 
                       help="Up block types for UNet")
    parser.add_argument("--attention_head_dim", type=int, default=8, help="Number of attention heads")
    
    # Training args
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr_warmup_percentage", type=float, default=0.05, help="Learning rate warmup portion") 
    parser.add_argument("--lr_scheduler_cycles", type=float, default=0.5, help="Number of cycles for the cosine learning rate scheduler")
    parser.add_argument("--save_image_epochs", type=int, default=20, help="Save generated levels every N epochs")
    parser.add_argument("--save_model_epochs", type=int, default=20, help="Save model every N epochs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate_epochs", type=int, default=5, help="Calculate validation loss every N epochs")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="level-diffusion-output", help="Output directory")
    
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
    
    if args.negative_prompt_training and not args.text_conditional:
        raise ValueError("Negative prompt training requires text conditioning to be enabled")

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
    if args.text_conditional and args.mlm_model_dir:
        text_encoder = TransformerModel.from_pretrained(args.mlm_model_dir).to(device)
        text_encoder.eval()  # Set to evaluation mode
        print(f"Loaded text encoder from {args.mlm_model_dir}")
    
    # Initialize dataset
    if args.split:
        train_json, val_json, test_json = split_dataset(args.json, args.train_pct, args.val_pct, args.test_pct)
        train_dataset = LevelDataset(
            json_path=train_json,
            tokenizer=tokenizer,
            shuffle=True,
            mode="diffusion",
            augment=args.augment,
            num_tiles=args.num_tiles,
            negative_captions=args.negative_prompt_training
        )
        val_dataset = LevelDataset(
            json_path=val_json,
            tokenizer=tokenizer,
            shuffle=False,
            mode="diffusion",
            augment=False,
            num_tiles=args.num_tiles,
            negative_captions=args.negative_prompt_training
        )
    else:
        train_dataset = LevelDataset(
            json_path=args.json,
            tokenizer=tokenizer,
            shuffle=True,
            mode="diffusion",
            augment=args.augment,
            num_tiles=args.num_tiles,
            negative_captions=args.negative_prompt_training
        )
        val_dataset = None

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True
        )

    if args.text_conditional:
        # Sample four random captions from the dataset
        sample_indices = [random.randint(0, len(train_dataset) - 1) for _ in range(4)]
        if args.negative_prompt_training:
            sample_data = [train_dataset[i] for i in sample_indices]
            pos_vectors = [data[1] for data in sample_data]
            neg_vectors = [data[2] for data in sample_data]
            pos_vectors = [v.tolist() for v in pos_vectors]
            neg_vectors = [v.tolist() for v in neg_vectors]
            pad_token = tokenizer.token_to_id["[PAD]"]
            sample_captions = [
                tokenizer.decode([token for token in caption if token != pad_token]) 
                for caption in pos_vectors
            ]
            sample_negative_captions = [
                tokenizer.decode([token for token in caption if token != pad_token]) 
                for caption in neg_vectors
            ]
            print("Sample positive captions:")
            for caption in sample_captions:
                print(f"  POS: {caption}")
            print("Sample negative captions:")
            for caption in sample_negative_captions:
                print(f"  NEG: {caption}")
        else:
            # Original code for positive-only captions
            sample_embedding_vectors = [train_dataset[i][1] for i in sample_indices]
            sample_embedding_vectors = [v.tolist() for v in sample_embedding_vectors]
            pad_token = tokenizer.token_to_id["[PAD]"]
            sample_captions = [
                tokenizer.decode([token for token in caption if token != pad_token]) 
                for caption in sample_embedding_vectors
            ]
            print("Sample captions:")
            for caption in sample_captions:
                print(caption)

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
            cross_attention_dim=text_encoder.embedding_dim,  # Match the embedding dimension
            attention_head_dim=args.attention_head_dim,  # Number of attention heads
        )
        # Add flag for negative prompt support if enabled
        if args.negative_prompt_training:
            model.negative_prompt_support = True
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
    total_training_steps = (len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_training_steps * args.lr_warmup_percentage)  

    print(f"Warmup period will be {warmup_steps} steps out of {total_training_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_cycles=args.lr_scheduler_cycles,
        num_warmup_steps=warmup_steps,  # Use calculated warmup steps
        num_training_steps=total_training_steps,
    )
    
    # Prepare for training with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(total=args.num_epochs * len(train_dataloader), disable=not accelerator.is_local_main_process)
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
    def log_metrics(epoch, loss, lr, step=None, val_loss=None):
        if accelerator.is_local_main_process:
            log_entry = {
                "epoch": epoch,
                "loss": loss,
                "lr": lr,
                "step": step if step is not None else epoch * len(train_dataloader),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if val_loss is not None:
                log_entry["val_loss"] = val_loss
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    # Initialize plotter if we're on the main process
    plotter = None
    plot_thread = None
    if accelerator.is_local_main_process:
        plotter = LossPlotter(log_file, update_interval=5.0, left_key='loss', right_key='val_loss',
                             left_label='Training Loss', right_label='Validation Loss')
        plot_thread = threading.Thread(target=plotter.start_plotting)
        plot_thread.daemon = True
        plot_thread.start()
        print(f"Loss plotting enabled. Progress will be saved to {os.path.join(args.output_dir, 'training_progress.png')}")

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_dataloader):

            # Process batch data
            if args.text_conditional:
                # Unpack scenes and captions
                if args.negative_prompt_training:
                    scenes, captions, negative_captions = batch
                else:
                    scenes, captions = batch
    
                # First generate timesteps before we duplicate anything
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (scenes.shape[0],), device=scenes.device).long()

                # Get text embeddings from the text encoder
                with torch.no_grad():
                    text_embeddings = text_encoder.get_embeddings(captions)
                    if args.negative_prompt_training:
                        negative_embeddings = text_encoder.get_embeddings(negative_captions)
                        # For negative prompt training, we use three sets of embeddings:
                        # [negative_embeddings, uncond_embeddings, text_embeddings]
                        uncond_tokens = torch.zeros_like(captions)
                        uncond_embeddings = text_encoder.get_embeddings(uncond_tokens)
                        combined_embeddings = torch.cat([negative_embeddings, uncond_embeddings, text_embeddings])
                        scenes_for_train = torch.cat([scenes] * 3)  # Repeat scenes three times
                        timesteps_for_train = torch.cat([timesteps] * 3)  # Repeat timesteps three times
                    else:
                        # Original classifier-free guidance with just uncond and cond
                        uncond_tokens = torch.zeros_like(captions)
                        uncond_embeddings = text_encoder.get_embeddings(uncond_tokens)
                        combined_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                        scenes_for_train = torch.cat([scenes] * 2)  # Repeat scenes twice
                        timesteps_for_train = torch.cat([timesteps] * 2)  # Repeat timesteps twice
    
                # Add noise to the clean scenes
                noise = torch.randn_like(scenes_for_train)
                noisy_scenes = noise_scheduler.add_noise(scenes_for_train, noise, timesteps_for_train)
    
                with accelerator.accumulate(model):
                    # Predict the noise with conditioning
                    noise_pred = model(noisy_scenes, timesteps_for_train, encoder_hidden_states=combined_embeddings).sample
        
                    # Compute loss
                    loss = F.mse_loss(noise_pred, noise)
        
                    # Backpropagation
                    accelerator.backward(loss)
        
                    # Update the model parameters
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            else:
                # For unconditional generation, we don't need captions
                if isinstance(batch, list):
                    scenes, _ = batch  # Ignore captions
                else:
                    scenes = batch

                # Add noise to the clean scenes
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (scenes.shape[0],), device=scenes.device).long()
                noise = torch.randn_like(scenes)
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
            
            train_loss += loss.detach().item()
            
            # Update progress bar
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)
                        
            global_step += 1
        
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Calculate validation loss if validation dataset exists and it's time to validate
        val_loss = None
        if val_dataloader is not None and (epoch % args.validate_epochs == 0 or epoch == args.num_epochs - 1):
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    if args.text_conditional:
                        if args.negative_prompt_training:
                            val_scenes, val_captions, val_negative_captions = val_batch
                            val_scenes = val_scenes.to(device)
                            val_captions = val_captions.to(device)
                            val_negative_captions = val_negative_captions.to(device)
                        else:
                            val_scenes, val_captions = val_batch
                            val_scenes = val_scenes.to(device)
                            val_captions = val_captions.to(device)
                            
                        val_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                                    (val_scenes.shape[0],), device=device).long()
                        
                        val_noise = torch.randn_like(val_scenes)
                        val_noisy_scenes = noise_scheduler.add_noise(val_scenes, val_noise, val_timesteps)
                        
                        with torch.no_grad():
                            val_text_embeddings = text_encoder.get_embeddings(val_captions)
                            if args.negative_prompt_training:
                                val_negative_embeddings = text_encoder.get_embeddings(val_negative_captions)
                                val_uncond_tokens = torch.zeros_like(val_captions)
                                val_uncond_embeddings = text_encoder.get_embeddings(val_uncond_tokens)
                                val_combined_embeddings = torch.cat([val_negative_embeddings, val_uncond_embeddings, val_text_embeddings])
                                val_scenes_for_eval = torch.cat([val_scenes] * 3)
                                val_timesteps_for_eval = torch.cat([val_timesteps] * 3)
                            else:
                                val_uncond_tokens = torch.zeros_like(val_captions)
                                val_uncond_embeddings = text_encoder.get_embeddings(val_uncond_tokens)
                                val_combined_embeddings = torch.cat([val_uncond_embeddings, val_text_embeddings])
                                val_scenes_for_eval = torch.cat([val_scenes] * 2)
                                val_timesteps_for_eval = torch.cat([val_timesteps] * 2)
                                
                        val_noise_pred = model(val_scenes_for_eval, val_timesteps_for_eval, 
                                            encoder_hidden_states=val_combined_embeddings).sample
                        val_batch_loss = F.mse_loss(val_noise_pred, torch.cat([val_noise] * (3 if args.negative_prompt_training else 2)))
                    else:
                        if isinstance(val_batch, list):
                            val_scenes, _ = val_batch
                        else:
                            val_scenes = val_batch
                        val_scenes = val_scenes.to(device)
                            
                        val_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                                    (val_scenes.shape[0],), device=device).long()
                        val_noise = torch.randn_like(val_scenes)
                        val_noisy_scenes = noise_scheduler.add_noise(val_scenes, val_noise, val_timesteps)
                        val_noise_pred = model(val_noisy_scenes, val_timesteps).sample
                        val_batch_loss = F.mse_loss(val_noise_pred, val_noise)
                        
                    val_loss += val_batch_loss.item()
                    
            val_loss /= len(val_dataloader)
            model.train()
            
        # Log metrics including validation loss
        log_metrics(epoch, avg_train_loss, lr_scheduler.get_last_lr()[0], val_loss=val_loss, step=global_step)
        
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
                ).to("cuda")
                                
                # Use the raw negative captions instead of tokens
                with torch.no_grad():
                    samples = pipeline(
                        batch_size=4,
                        generator=torch.Generator(device=accelerator.device).manual_seed(args.seed),
                        num_inference_steps = 50, # Fewer steps needed for inference
                        output_type="tensor",
                        caption=sample_captions,
                        negative_prompt=sample_negative_captions if args.negative_prompt_training else None 
                    ).images
            else:
                # For unconditional generation
                pipeline = UnconditionalDDPMPipeline(
                    unet=accelerator.unwrap_model(model), 
                    scheduler=noise_scheduler
                )
                
                # Generate sample levels
                with torch.no_grad():
                    samples = pipeline(
                        batch_size=4,
                        generator=torch.Generator(device=accelerator.device).manual_seed(args.seed),
                        num_inference_steps = 50, # Fewer steps needed for inference
                        output_type="tensor",
                    ).images

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
                ).to("cuda")
                # Save negative prompt support flag if enabled
                if args.negative_prompt_training:
                    pipeline.supports_negative_prompt = True
            else:
                pipeline = UnconditionalDDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                
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
        ).to("cuda")
    else:
        pipeline = UnconditionalDDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
        
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

def split_dataset(json_path, train_pct, val_pct, test_pct):
    """Splits the dataset into train/val/test and saves them as new JSON files."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    n = len(data)
    indices = list(range(n))
    random.shuffle(indices)
    train_end = int(train_pct * n)
    val_end = train_end + int(val_pct * n)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
    base, ext = os.path.splitext(json_path)
    train_path = f"{base}-train{ext}"
    val_path = f"{base}-validate{ext}"
    test_path = f"{base}-test{ext}"
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    return train_path, val_path, test_path

if __name__ == "__main__":
    main()
