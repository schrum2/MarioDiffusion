# Completely different attempt at training the diffusion model from Claude

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import UNet2DModel, DDPMScheduler
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

# From me
from level_dataset import LevelDataset, visualize_samples
from tokenizer import Tokenizer 
from torch.utils.data import DataLoader

class TileDiffusionTrainer:
    def __init__(
        self,
        num_tiles=15,
        image_size=16,
        batch_size=32,
        learning_rate=1e-4,
        num_train_timesteps=1000,
        num_inference_steps=100,
        beta_schedule="cosine",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.num_tiles = num_tiles
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Create UNet model
        self.model = UNet2DModel(
            sample_size=image_size,           # The size of the generated images (16x16)
            in_channels=num_tiles,            # Number of input channels (15 tile types)
            out_channels=num_tiles,           # Output the same number of channels
            layers_per_block=2,               # Number of ResNet blocks per downsample
            block_out_channels=(128, 256, 512, 512),  # Number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D", 
                "AttnDownBlock2D",  # Add attention layers for better spatial correlations
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",    # Matching attention in upsampling path
                "UpBlock2D",
                "UpBlock2D"
            ),
        ).to(device)
        
        # Create noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,       # "cosine" schedule for better handling of discrete data
            prediction_type="epsilon",         # Predict the noise to be removed
            clip_sample=False,                 # Don't clip the values
        )
        
        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # For inference
        self.num_inference_steps = num_inference_steps
    
    def train(self, dataloader, num_epochs=100, save_interval=10, save_path="tile_diffusion_model.pt"):
        """Train the diffusion model"""
        
        # Set up tracking for losses
        epoch_losses = []
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for batch_idx, batch in enumerate(progress_bar):
                scenes, _ = batch  # Ignore captions
                scenes = scenes.to(self.device)
                
                # Sample noise to add to the images
                noise = torch.randn_like(scenes).to(self.device)
                
                # Sample a random timestep for each image
                batch_size = scenes.shape[0]
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, 
                    (batch_size,), device=self.device
                ).long()
                
                # Add noise to the clean images according to the noise magnitude at each timestep
                # (Forward diffusion process)
                noisy_images = self.noise_scheduler.add_noise(scenes, noise, timesteps)
                
                # Get the model prediction for the noise
                noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
                
                # Calculate the loss
                loss = F.mse_loss(noise_pred, noise)
                
                # Update model parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.6f}")
            
            # Save model checkpoint
            if (epoch + 1) % save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                }, f"{save_path.split('.')[0]}_epoch{epoch+1}.pt")
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                }, save_path)
                print(f"New best model saved with loss: {best_loss:.6f}")
        
        # Plot training curve
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('training_loss.png')
        plt.close()
        
        return epoch_losses
    
    def generate_samples(self, num_samples=1):
        """Generate new samples using the trained diffusion model"""
        self.model.eval()
        
        with torch.no_grad():
            # Start with random noise
            sample = torch.randn(
                (num_samples, self.num_tiles, self.image_size, self.image_size),
                device=self.device
            )
            
            # Set up inference scheduler
            inference_scheduler = DDPMScheduler(
                num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
                beta_schedule=self.noise_scheduler.config.beta_schedule,
                clip_sample=False,
                prediction_type="epsilon",
            )
            
            # Sampling loop
            for t in tqdm(inference_scheduler.timesteps[-self.num_inference_steps:]):
                # Get model prediction
                model_output = self.model(sample, t.unsqueeze(0).repeat(num_samples), return_dict=False)[0]
                
                # Update sample with scheduler
                sample = inference_scheduler.step(model_output, t, sample).prev_sample
            
            # Convert continuous output to one-hot encoding
            # For each position, find the tile with highest probability
            sample = F.softmax(sample, dim=1)
            
            return sample
    
    def convert_to_categorical(self, samples):
        """Convert softmax probabilities to hard categories"""
        # Get the index of the highest probability for each position
        indices = torch.argmax(samples, dim=1)
        
        # Create one-hot encoding
        batch_size = samples.shape[0]
        categorical = torch.zeros(
            (batch_size, self.num_tiles, self.image_size, self.image_size),
            device=samples.device
        )
        
        # Set the appropriate values to 1
        for i in range(batch_size):
            for x in range(self.image_size):
                for y in range(self.image_size):
                    tile_idx = indices[i, x, y].item()
                    categorical[i, tile_idx, x, y] = 1.0
        
        return categorical

# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a diffusion model for tile-based level generation")
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset JSON file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model every N epochs")
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer.load(args.pkl)

    # Initialize dataset and dataloader (as in your example)
    dataset = LevelDataset(
        json_path=args.json,
        tokenizer=tokenizer,  # You'll need to define this
        batch_size=args.batch_size,
        shuffle=True,
        mode="diffusion",
        augment=args.augment,
        num_tiles=args.num_tiles
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    # Initialize and train the diffusion model
    trainer = TileDiffusionTrainer(
        num_tiles=args.num_tiles,
        image_size=16,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Train the model
    losses = trainer.train(
        dataloader, 
        num_epochs=args.epochs, 
        save_interval=args.save_interval
    )
    
    # Generate some samples
    samples = trainer.generate_samples(num_samples=4)
    categorical_samples = trainer.convert_to_categorical(samples)
    
    print("Generation complete. Sample shapes:", categorical_samples.shape)