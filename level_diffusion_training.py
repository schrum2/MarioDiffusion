import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
from tokenizer import Tokenizer 


def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model for level generation")
    parser.add_argument("--json_path", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to JSON file with level data")
    parser.add_argument("--tokenizer_path", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model")
    parser.add_argument("--mlm_model_path", type=str, default="mlm_transformer.pth", help="Path to trained MLM model (for conditional training)")
    parser.add_argument("--conditional", action="store_true", help="Enable text conditional training")
    parser.add_argument("--num_train_steps", type=int, default=100000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save model every X steps")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    parser.add_argument("--level_size", type=int, default=16, help="Size of level (assumed square)")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA for model weights")
    return parser.parse_args()

class TextEncoder(nn.Module):
    def __init__(self, transformer_model, embedding_dim):
        super(TextEncoder, self).__init__()
        self.transformer = transformer_model
        # Add projection if needed to match diffusion model dimensions
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x):
        # Get transformer output
        transformer_output = self.transformer(x)
        # Use mean pooling for a single embedding vector
        # Exclude padding tokens by creating a mask
        padding_mask = (x != 0).float()  # Assuming 0 is the padding token ID
        # Apply mask and calculate mean
        masked_output = transformer_output * padding_mask.unsqueeze(-1)
        sum_embeddings = masked_output.sum(dim=1)
        sum_mask = padding_mask.sum(dim=1).unsqueeze(-1)
        mean_embeddings = sum_embeddings / (sum_mask + 1e-9)
        # Project to required dimension
        projected = self.projection(mean_embeddings)
        return projected


def create_model(args):
    # For unconditional model
    if not args.conditional:
        model = UNet2DModel(
            sample_size=args.level_size,  # 16x16 levels
            in_channels=args.num_tiles,   # One channel per tile type
            out_channels=args.num_tiles,  # One channel per tile type
            layers_per_block=2,           # Number of resnet blocks per downsample
            block_out_channels=(128, 256, 512, 512),  # Channel size for each resnet block
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    else:
        # For conditional model - will implement text conditioning
        # Import the MLM transformer
        from level_transformer import TransformerModel
        
        # Load trained MLM model
        checkpoint = torch.load(args.mlm_model_path)
        vocab_size = checkpoint["vocab_size"]
        embedding_dim = checkpoint["embedding_dim"]
        hidden_dim = checkpoint["hidden_dim"]
        
        # Create and load transformer
        transformer = TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        transformer.load_state_dict(checkpoint["model_state_dict"])
        
        # Create text encoder
        text_encoder = TextEncoder(transformer, embedding_dim)
        
        # Create conditional UNet
        model = UNet2DModel(
            sample_size=args.level_size,
            in_channels=args.num_tiles,
            out_channels=args.num_tiles,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            # Add cross-attention parameters
            cross_attention_dim=embedding_dim,
        )
        
        return model, text_encoder
    
    return model


def train_diffusion_model(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer.load(args.tokenizer_path)
    
    # Create dataset
    from level_dataset import LevelDataset
    diffusion_dataset = LevelDataset(
        json_path=args.json_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        mode="diffusion",
        augment=args.augment,
        num_tiles=args.num_tiles
    )

    # Setup noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
    )
    
    # Create model
    if args.conditional:
        model, text_encoder = create_model(args)
    else:
        model = create_model(args)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if args.conditional:
        text_encoder.to(device)
        text_encoder.eval()  # Freeze the text encoder during training
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.num_train_steps,
    )
    
    # Setup EMA
    if args.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.9999)
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(range(args.num_train_steps))
    
    while global_step < args.num_train_steps:
        for batch_idx in range(len(diffusion_dataset)):
            # Get batch
            scenes, captions = diffusion_dataset[batch_idx]
            clean_images = scenes.to(device)
            
            # Sample noise
            noise = torch.randn_like(clean_images)
            batch_size = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            ).long()
            
            # Add noise to images according to timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Conditional model branch
            if args.conditional:
                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(captions.to(device))
                
                # Predict noise
                noise_pred = model(
                    noisy_images, 
                    timesteps, 
                    encoder_hidden_states=encoder_hidden_states
                ).sample
            else:
                # Unconditional model branch
                noise_pred = model(noisy_images, timesteps).sample
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Update EMA
            if args.use_ema:
                ema_model.step(model.parameters())
            
            # Log progress
            progress_bar.update(1)
            global_step += 1
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            
            # Save model checkpoint
            if global_step % args.save_steps == 0 or global_step == args.num_train_steps:
                # Create pipeline
                if args.use_ema:
                    # Get EMA model parameters
                    ema_params = EMAModel.from_state_dict(ema_model.averaged_params)
                    pipeline = DDPMPipeline(
                        unet=model,
                        scheduler=noise_scheduler,
                    )
                    # Set EMA parameters
                    pipeline.unet.load_state_dict(ema_params)
                else:
                    pipeline = DDPMPipeline(
                        unet=model,
                        scheduler=noise_scheduler,
                    )
                
                # Save pipeline
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                pipeline.save_pretrained(save_dir)
                
                # Save other components if conditional
                if args.conditional:
                    torch.save({
                        "text_encoder": text_encoder.state_dict(),
                    }, os.path.join(save_dir, "text_encoder.pt"))
            
            # Check if we've reached the total steps
            if global_step >= args.num_train_steps:
                break
    
    # Save final model
    if args.use_ema:
        ema_params = EMAModel.from_state_dict(ema_model.averaged_params)
        pipeline = DDPMPipeline(
            unet=model,
            scheduler=noise_scheduler,
        )
        pipeline.unet.load_state_dict(ema_params)
    else:
        pipeline = DDPMPipeline(
            unet=model,
            scheduler=noise_scheduler,
        )
    
    # Save final pipeline
    pipeline.save_pretrained(args.output_dir)
    
    # Save conditional components if needed
    if args.conditional:
        torch.save({
            "text_encoder": text_encoder.state_dict(),
        }, os.path.join(args.output_dir, "text_encoder.pt"))
    
    return pipeline


def generate_level(
    pipeline, 
    num_tiles=15, 
    level_size=16, 
    num_inference_steps=1000, 
    guidance_scale=7.5,
    text_encoder=None, 
    prompt=None,
    device="cuda",
):
    """Generate a level using the trained diffusion model"""
    
    # Check if conditional generation is possible
    conditional = text_encoder is not None and prompt is not None
    
    # Generate initial noise
    sample_shape = (1, num_tiles, level_size, level_size)
    noise = torch.randn(sample_shape, device=device)
    
    # Run diffusion process
    if conditional:
        # Encode prompt
        prompt_embeds = text_encoder(prompt)
        
        # Setup classifier-free guidance
        if guidance_scale > 1.0:
            uncond_embeds = torch.zeros_like(prompt_embeds)
            prompt_embeds = torch.cat([uncond_embeds, prompt_embeds])
            
        # Generate image with guidance
        sample = pipeline(
            noise,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            encoder_hidden_states=prompt_embeds,
        ).images
    else:
        # Generate image without conditioning
        sample = pipeline(
            noise,
            num_inference_steps=num_inference_steps,
        ).images
    
    # Convert continuous probability distribution to discrete tiles
    # The output is one-hot encoded, so we take the argmax along the channel dimension
    level = torch.argmax(sample[0], dim=0).cpu()
    
    return level


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train diffusion model
    pipeline = train_diffusion_model(args)
    
    # Generate a sample level
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    level = generate_level(
        pipeline, 
        num_tiles=args.num_tiles, 
        level_size=args.level_size,
        device=device,
    )
    
    print("Generated level shape:", level.shape)
    print("Sample level:")
    print(level)


if __name__ == "__main__":
    main()