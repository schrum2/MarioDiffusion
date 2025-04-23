import torch
import torch.nn.functional as F
from typing import NamedTuple
import os
from diffusers import DDPMPipeline, UNet2DConditionModel, DDPMScheduler
from models.text_model import TransformerModel  
            
class PipelineOutput(NamedTuple):
    images: torch.Tensor

# Create a custom pipeline for text-conditional generation
class TextConditionalDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, text_encoder=None):
        super().__init__(unet=unet, scheduler=scheduler)
        self.text_encoder = text_encoder

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
        
    def __call__(self, batch_size=1, generator=None, num_inference_steps=1000, 
                output_type="tensor", captions=None, guidance_scale=7.5, 
                height=16, width=16, raw_latent_sample=None, input_scene=None, **kwargs):
        """
            Neither raw_latent_sample nor input_scene are needed, in which case
            random latent noise is the starting point for the diffusion. If 
            raw_latent_sample is provided, it is taken as the diffusion starting
            point. This means raw_latent_sample should have a shape that matches
            the unet (correct number of channels).
        """


        # Process text embeddings if captions are provided
        if captions is not None and self.text_encoder is not None:
            # Conditional embeddings from provided captions
            text_embeddings = self.text_encoder.get_embeddings(captions)
        
            # Unconditional embeddings for classifier-free guidance
            # Use empty/zero tokens for the unconditional case
            uncond_tokens = torch.zeros_like(captions)
            uncond_embeddings = self.text_encoder.get_embeddings(uncond_tokens)
        
            # Concatenate unconditional and conditional embeddings for CFG
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        elif self.text_encoder is not None:
            # For unconditional generation, we still need embeddings
            embedding_dim = self.text_encoder.embedding_dim
            seq_length = 10  # Any reasonable sequence length
            text_embeddings = torch.zeros(batch_size, seq_length, embedding_dim, device=self.device)
        else:
            raise ValueError("Text encoder needed")
    
        # Start from random noise
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("Must provide a generator for each sample if passing a list of generators")
        
        device = self.device
        sample_shape = (batch_size, self.unet.config.in_channels, height, width)
    
        if raw_latent_sample != None:
            sample = raw_latent_sample
            if sample.shape[1] != sample_shape[1]:
                raise ValueError(f"Wrong number of channels in raw_latent_sample: Expected {self.unet.config.in_channels} but sample had {sample.shape[1]} channels")
        elif isinstance(generator, list):
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
            # For classifier-free guidance, we need to do two forward passes:
            # one with the unconditional embedding and one with conditional embedding
        
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([sample] * 2) if captions is not None else sample
        
            # Prepare model inputs
            model_kwargs = {}
            if text_embeddings is not None:
                model_kwargs["encoder_hidden_states"] = text_embeddings
            
            # Predict noise residual
            noise_pred = self.unet(latent_model_input, t, **model_kwargs).sample
        
            # Perform guidance if using text conditioning
            if captions is not None:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # Combine predictions with classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
            # Compute previous sample: x_{t-1} = scheduler(x_t, noise_pred)
            sample = self.scheduler.step(noise_pred, t, sample).prev_sample
        
        # Convert to output format
        if output_type == "tensor":
            # Apply softmax to get probabilities for each tile type
            sample = F.softmax(sample, dim=1)
        else:
            raise ValueError(f"Unsupported output type: {output_type}")
        
        return PipelineOutput(images=sample)
