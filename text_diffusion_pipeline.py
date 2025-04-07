import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler, DDPMPipeline

class PipelineOutput(NamedTuple):
    images: torch.Tensor

# Create a custom pipeline for text-conditional generation
class TextConditionalDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, text_encoder=None):
        super().__init__(unet, scheduler)
        self.text_encoder = text_encoder

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        super().save_pretrained(save_directory)  # saves UNet and scheduler

        # Save custom text encoder
        if self.text_encoder is not None:
            self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))

    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        # Load unet and scheduler as usual
        pipeline = super().from_pretrained(pretrained_model_path, **kwargs)

        # Load the custom text encoder
        text_encoder_path = os.path.join(pretrained_model_path, "text_encoder")
        if os.path.exists(text_encoder_path):
            from transformer_model import TransformerModel  # import your custom model
            text_encoder = TransformerModel.from_pretrained(text_encoder_path)
        else:
            text_encoder = None

        return cls(
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            text_encoder=text_encoder
        )
        
    def __call__(self, batch_size=1, generator=None, num_inference_steps=1000, 
                output_type="pil", captions=None, **kwargs):
        # Process text embeddings if captions are provided
        text_embeddings = None
        if captions is not None and self.text_encoder is not None:
            text_embeddings = self.text_encoder.get_embeddings(captions)
            #print(text_embeddings.shape)
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
            
        return PipelineOutput(images=sample)
