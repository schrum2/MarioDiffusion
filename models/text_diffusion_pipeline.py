import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional
import os
from diffusers import DDPMPipeline, UNet2DConditionModel, DDPMScheduler
# Running the main at the end of this requires messing with this import
from models.text_model import TransformerModel  
            
class PipelineOutput(NamedTuple):
    images: torch.Tensor

# Create a custom pipeline for text-conditional generation
class TextConditionalDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, text_encoder=None):
        super().__init__(unet=unet, scheduler=scheduler)
        self.text_encoder = text_encoder
        self.supports_negative_prompt = hasattr(unet, 'negative_prompt_support') and unet.negative_prompt_support

        # Register the text_encoder so that .to(), .cpu(), .cuda(), etc. work correctly
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            text_encoder=text_encoder,
        )
    
    # Override the to() method to ensure text_encoder is moved to the correct device
    def to(self, device=None, dtype=None):
        # Call the parent's to() method first
        pipeline = super().to(device, dtype)
        
        # Additionally move the text_encoder to the device
        if self.text_encoder is not None:
            self.text_encoder.to(device)
        
        return pipeline

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        super().save_pretrained(save_directory)  # saves UNet and scheduler

        # Save custom text encoder
        if self.text_encoder is not None:
            self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))

    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        #from diffusers.utils import load_config, load_state_dict
        # Load model_index.json
        #model_index = load_config(pretrained_model_path)

        # Load components manually
        unet_path = os.path.join(pretrained_model_path, "unet")
        unet = UNet2DConditionModel.from_pretrained(unet_path)

        scheduler_path = os.path.join(pretrained_model_path, "scheduler")
        # Have heard that DDIMScheduler might be faster for inference, though not necessarily better
        scheduler = DDPMScheduler.from_pretrained(scheduler_path)

        text_encoder_path = os.path.join(pretrained_model_path, "text_encoder")
        if os.path.exists(text_encoder_path):
            text_encoder = TransformerModel.from_pretrained(text_encoder_path)
        else:
            text_encoder = None

        # Instantiate your pipeline
        pipeline = cls(
            unet=unet,
            scheduler=scheduler,
            text_encoder=text_encoder,
            **kwargs,
        )

        return pipeline
        
    def __call__(
        self,
        caption: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 1000,
        guidance_scale: float = 7.5,
        height: int = 16,
        width: int = 16,
        raw_latent_sample: Optional[torch.FloatTensor] = None,
        input_scene: Optional[torch.Tensor] = None,
        output_type: str = "tensor"
    ) -> PipelineOutput:
        """Generate an image based on text input using the diffusion model.

        Args:
            caption: Text description of the desired output. If None, generates unconditionally.
            negative_prompt: Text description of what should not appear in the output.
                Only works with models trained with negative prompt support.
            generator: Random number generator for reproducibility.
            num_inference_steps: Number of denoising steps (more = higher quality, slower).
            guidance_scale: How strongly the generation follows the text prompt (higher = stronger).
            height: Height of generated image in tiles.
            width: Width of generated image in tiles.
            raw_latent_sample: Optional starting point for diffusion instead of random noise.
                Must have correct number of channels matching the UNet.
            input_scene: Optional 2D int tensor where each value corresponds to a tile type.
                Will be converted to one-hot encoding as starting point.
            output_type: Currently only "tensor" is supported.

        Returns:
            PipelineOutput containing the generated image tensor.
        """
        # Validate text encoder if we need it
        if caption is not None and self.text_encoder is None:
            raise ValueError("Text encoder is required for conditional generation")

        self.unet.eval()
        if self.text_encoder is not None:
            self.text_encoder.to(self.device)
            self.text_encoder.eval()

        with torch.no_grad():
            # Process text embeddings if caption is provided
            if caption is not None:
                # Get embeddings for the caption
                caption_ids = self.text_encoder.tokenizer.encode(caption)
                caption_ids = torch.tensor([caption_ids], device=self.device)
                caption_embedding = self.text_encoder.get_embeddings(caption_ids)

                # Handle negative prompt if provided
                if negative_prompt is not None:
                    if not self.supports_negative_prompt:
                        raise ValueError("This model was not trained with negative prompt support")
                    
                    # Get embeddings for negative prompt
                    negative_ids = self.text_encoder.tokenizer.encode(negative_prompt)
                    negative_ids = torch.tensor([negative_ids], device=self.device)
                    negative_embedding = self.text_encoder.get_embeddings(negative_ids)
                    
                    # Get unconditional (empty) embedding
                    empty_ids = torch.zeros_like(caption_ids)
                    empty_embedding = self.text_encoder.get_embeddings(empty_ids)
                    
                    # Concatenate [negative, unconditional, conditional] embeddings
                    text_embeddings = torch.cat([negative_embedding, empty_embedding, caption_embedding])
                else:
                    # Standard classifier-free guidance with just [unconditional, conditional]
                    empty_ids = torch.zeros_like(caption_ids)
                    empty_embedding = self.text_encoder.get_embeddings(empty_ids)
                    text_embeddings = torch.cat([empty_embedding, caption_embedding])
            
            else:
                # For unconditional generation, we still need empty embeddings
                seq_length = 10  # Any reasonable sequence length
                empty_ids = torch.zeros((1, seq_length), dtype=torch.long, device=self.device)
                text_embeddings = self.text_encoder.get_embeddings(empty_ids)

            #print(text_embeddings.shape)

            # Set up initial latent state
            device = self.device
            sample_shape = (1, self.unet.config.in_channels, height, width)
            
            if raw_latent_sample is not None:
                if input_scene is not None:
                    raise ValueError("Cannot provide both raw_latent_sample and input_scene")
                
                sample = raw_latent_sample.to(device)
                if sample.shape[1] != sample_shape[1]:
                    raise ValueError(f"Wrong number of channels in raw_latent_sample: Expected {self.unet.config.in_channels} but got {sample.shape[1]}")
            
            elif input_scene is not None:
                # Convert input scene to one-hot encoding
                scene_tensor = torch.tensor(input_scene, dtype=torch.long, device=device)
                one_hot = F.one_hot(scene_tensor, num_classes=self.unet.config.in_channels).float()
                # Reshape to (1, channels, height, width)
                sample = one_hot.permute(2, 0, 1).unsqueeze(0)
            
            else:
                # Start from random noise
                sample = torch.randn(sample_shape, generator=generator, device=device)
            
            # Set up diffusion process
            self.scheduler.set_timesteps(num_inference_steps)
            
            # Denoising loop
            for t in self.progress_bar(self.scheduler.timesteps):
                # Handle conditional generation
                if caption is not None:
                    if negative_prompt is not None:
                        # Three copies for negative prompt guidance
                        model_input = torch.cat([sample] * 3)
                    else:
                        # Two copies for standard classifier-free guidance
                        model_input = torch.cat([sample] * 2)
                else:
                    model_input = sample

                # Predict noise residual
                model_kwargs = {"encoder_hidden_states": text_embeddings} 
                noise_pred = self.unet(model_input, t, **model_kwargs).sample
                
                # Apply guidance
                if caption is not None:
                    if negative_prompt is not None:
                        # Split predictions for negative, unconditional, and text-conditional
                        noise_pred_neg, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                        # Apply guidance away from negative and towards conditional
                        noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        noise_pred = noise_pred_guided - guidance_scale * (noise_pred_neg - noise_pred_uncond)
                    else:
                        # Standard classifier-free guidance
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute previous sample: x_{t-1} = scheduler(x_t, noise_pred)
                sample = self.scheduler.step(noise_pred, t, sample, generator=generator).prev_sample
            
            # Convert to output format
            if output_type == "tensor":
                # Apply softmax to get probabilities for each tile type
                sample = F.softmax(sample, dim=1)
            else:
                raise ValueError(f"Unsupported output type: {output_type}")
        
        return PipelineOutput(images=sample)

    def print_unet_architecture(self):
        """Prints the architecture of the UNet model."""
        print(self.unet)

    def print_text_encoder_architecture(self):
        """Prints the architecture of the text encoder model, if it exists."""
        if self.text_encoder is not None:
            print(self.text_encoder)
        else:
            print("No text encoder is set.")

if __name__ == "__main__":
    import os
    import torch
    from level_dataset import visualize_samples

    # This won't run unless some imports at the top of the file are modified

    # Set up the pipeline
    model_path = "prev-cond-model"
    pipe = TextConditionalDDPMPipeline.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)

    # Generate with test prompt
    output = pipe(
        caption="full floor. two pipes.",
        num_inference_steps=50,
        guidance_scale=7.5,
        height=16,
        width=16,
    )

    # Convert output to proper format for visualization
    sample_images = visualize_samples(output.images, use_tiles=True)
    sample_images.show()

    # The visualize_samples function will save the image and also return the tile indices
    print("Generation complete! Check the generated image.")
