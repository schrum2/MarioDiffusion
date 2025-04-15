import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler, DDPMPipeline
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
from loss_plotter import LossPlotter
from models import TransformerModel
from text_diffusion_pipeline import TextConditionalDDPMPipeline
from level_dataset import visualize_samples, convert_to_level_format, samples_to_scenes
import json
import random
from text_diffusion_pipeline import TextConditionalDDPMPipeline
from create_ascii_captions import assign_caption, get_tile_descriptors, save_level_data
from caption_match import compare_captions

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate caption adherence for a pretrained text-conditional diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size") # TODO: Consider reducing to 16 to help generalization
        
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--inference_steps", type=int, default=500, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--save_as_json", action="store_true", help="Save generated levels as JSON")

    # Used to generate captions when generating images
    parser.add_argument("--tileset", default='..\TheVGLC\Super Mario Bros\smb.json', help="Descriptions of individual tile types")
    parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")

    # Output args
    parser.add_argument("--output_dir", type=str, default="text_to_level_samples", help="Output directory")
    
    return parser.parse_args()

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pipe = TextConditionalDDPMPipeline.from_pretrained(args.model_path).to(device)

    # Initialize dataset
    dataset = LevelDataset(
        json_path=args.json,
        tokenizer=pipe.text_encoder.tokenizer,
        shuffle=False,
        mode="diffusion",
        augment=False,
        num_tiles=args.num_tiles
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    with open(args.tileset, "r") as f:
        tileset = json.load(f)
        tile_chars = sorted(tileset['tiles'].keys())
        id_to_char = {idx: char for idx, char in enumerate(tile_chars)}
        char_to_id = {char: idx for idx, char in enumerate(tile_chars)}
        tile_descriptors = get_tile_descriptors(tileset)    
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        # Unpack scenes and captions
        scenes, captions = batch

        sample_caption_tokens = pipe.text_encoder.tokenizer.encode_batch(captions)
        sample_caption_tokens = torch.tensor(sample_caption_tokens).to(device)

        param_values = {
            "captions" : sample_caption_tokens,
            "num_inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            #"width": 16, # Might consider changing this later
            "output_type" : "tensor",
            "batch_size" : len(sample_caption_tokens)
        }
        generator = torch.Generator(device).manual_seed(int(args.seed))
        
        images = pipe(generator=generator, **param_values).images

        # Iterate over captions and corresponding generated images
        for caption, image in zip(captions, images):
            sample_tensor = image.unsqueeze(0)
            sample_indices = convert_to_level_format(sample_tensor)
            scene = sample_indices[0].tolist()  # Always just one scene: (1,16,16)
            actual_caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, args.describe_locations, args.describe_absence)

            compare_score = compare_captions(caption, actual_caption)
            # Optionally, log or save compare_score if needed






    


















    for epoch in range(args.num_epochs):
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):

            # Process batch data
            if args.text_conditional:
                # Unpack scenes and captions
                scenes, captions = batch
    
                # Get text embeddings from the text encoder
                with torch.no_grad():
                    text_embeddings = text_encoder.get_embeddings(captions)
    
                # For classifier-free guidance, we need unconditional embeddings
                uncond_tokens = torch.zeros_like(captions)
                with torch.no_grad():
                    uncond_embeddings = text_encoder.get_embeddings(uncond_tokens)
    
                # First generate timesteps before we duplicate anything
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (scenes.shape[0],), device=scenes.device).long()
    
                # Concatenate for training with classifier-free guidance
                # This way the model learns both conditional and unconditional generation
                batch_size = scenes.shape[0]
                scenes_for_train = torch.cat([scenes] * 2)  # Repeat scenes for both cond and uncond
                timesteps_for_train = torch.cat([timesteps] * 2)  # Repeat timesteps
                combined_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
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
                ).to("cuda")
                
                # Convert captions to token IDs using the tokenizer
                sample_caption_tokens = tokenizer.encode_batch(sample_captions)
                sample_caption_tokens = torch.tensor(sample_caption_tokens).to(accelerator.device)
                
                with torch.no_grad():
                    # Generate samples
                    samples = pipeline(
                        batch_size=4,
                        generator=torch.Generator(device=accelerator.device).manual_seed(args.seed),
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
                        generator=torch.Generator(device=accelerator.device).manual_seed(args.seed),
                        num_inference_steps=args.num_train_timesteps,
                        output_type="tensor",
                    ).images

                    # Seems odd that conditional model does not need this, but unconditional does
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
                ).to("cuda")
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
        ).to("cuda")
    else:
        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
        
    pipeline.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
