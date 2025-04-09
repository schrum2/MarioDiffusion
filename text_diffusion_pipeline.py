import torch
import torch.nn.functional as F
from typing import NamedTuple
import os
from diffusers import DDPMPipeline, UNet2DConditionModel, DDPMScheduler
from models import TransformerModel  
            
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
                output_type="tensor", captions=None, guidance_scale=7.5, **kwargs):
        
        # Create unconditional embeddings for classifier-free guidance
        # Use PAD tokens for the null/empty conditioning
        pad_token_id = self.text_encoder.tokenizer.token_to_id["[PAD]"]

        # Process text embeddings if captions are provided
        if captions is not None and self.text_encoder is not None:
            # Get text embeddings for conditioned generation
            text_embeddings = self.text_encoder.get_embeddings(captions)
            
            seq_length = text_embeddings.shape[1]  # Match the length of your conditional embeddings
            # Create sequences of PAD tokens
            pad_tokens = torch.full((batch_size, seq_length), pad_token_id, device=self.device)
            # Get embeddings for these PAD sequences
            uncond_embeddings = self.text_encoder(pad_tokens)  # Adjust based on your encoder's input format

            # Concatenate unconditional and conditional embeddings
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        elif self.text_encoder is not None:
            # Use empty prompts for unconditional generation
            embedding_dim = self.text_encoder.embedding_dim
            # This could be any number. It represents the length of the text caption.
            # but this random generation does not use any real tokens.
            seq_length = 10
            random_shape = (batch_size, seq_length, embedding_dim)
            text_embeddings = torch.randn(random_shape, device=self.device)
        else:
            raise ValueError("text encoder needed")        

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
            # Duplicate sample for classifier-free guidance
            model_input = torch.cat([sample] * 2) if captions is not None else sample

            model_kwargs = {}
            if text_embeddings is not None:
                model_kwargs["encoder_hidden_states"] = text_embeddings
                
            # Predict noise residual
            noise_pred = self.unet(model_input, t, **model_kwargs).sample
            
            # Perform guidance if using text conditioning
            if captions is not None:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample: x_{t-1} = scheduler(x_t, noise_pred)
            sample = self.scheduler.step(noise_pred, t, sample).prev_sample
            
        # Convert to output format
        if output_type == "tensor":
            # Apply softmax to get probabilities for each tile type
            sample = F.softmax(sample, dim=1)
        else:
            raise ValueError("Unsupported output type: {}".format(output_type))
            
        return PipelineOutput(images=sample)
    