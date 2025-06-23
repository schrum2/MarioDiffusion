from diffusers import DDPMPipeline
import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.ddpm.pipeline_ddpm import ImagePipelineOutput
import util.common_settings as common_settings
import os
import json

class UnconditionalDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, block_embeddings=None):
        super().__init__(unet, scheduler)

        self.block_embeddings = block_embeddings
        if self.block_embeddings is not None:
            self.using_block_embeds = True
        else:
            self.using_block_embeds = False
    

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        super().save_pretrained(save_directory)
        # Save using_block_embeds flag
        using_block_embeds = getattr(self, "using_block_embeds", False)
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            json.dump({"using_block_embeds": using_block_embeds}, f)
        # Save block_embeddings tensor if it exists
        if hasattr(self, "block_embeddings") and self.block_embeddings is not None:
            torch.save(self.block_embeddings, os.path.join(save_directory, "block_embeddings.pt"))

    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        pipeline = super().from_pretrained(pretrained_model_path, **kwargs)
        # Load using_block_embeds flag
        config_path = os.path.join(pretrained_model_path, "pipeline_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            setattr(pipeline, "using_block_embeds", config.get("using_block_embeds", False))
        else:
            setattr(pipeline, "using_block_embeds", False)
        # Load block_embeddings tensor if it exists
        block_embeds_path = os.path.join(pretrained_model_path, "block_embeddings.pt")
        if os.path.exists(block_embeds_path):
            pipeline.block_embeddings = torch.load(block_embeds_path, map_location="cpu")
        else:
            pipeline.block_embeddings = None
        return pipeline
    


    def give_sprite_scaling_factors(self, sprite_scaling_factors):
        """
        Set the sprite scaling factors for the pipeline.
        This is used to apply per-sprite temperature scaling during inference.
        """
        self.sprite_scaling_factors = sprite_scaling_factors

    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = common_settings.NUM_INFERENCE_STEPS,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        height: int = common_settings.MARIO_HEIGHT, width: int = common_settings.MARIO_WIDTH, 
        latents: Optional[torch.FloatTensor] = None,
        show_progress_bar=True,
    ) -> Union[ImagePipelineOutput, Tuple]:

        self.unet.eval()
        with torch.no_grad():

            if latents is not None:
                image = latents.to(self.device)
            else:
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels,
                    height,
                    width
                )

                image = torch.randn(image_shape, generator=generator, device=self.device)

            self.scheduler.set_timesteps(num_inference_steps)

            iterator = self.progress_bar(self.scheduler.timesteps) if show_progress_bar else self.scheduler.timesteps
            for t in iterator:
                #print(image.shape)
                model_output = self.unet(image, t).sample
                image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

            # Apply per-sprite temperature scaling if enabled
            if hasattr(self,"sprite_scaling_factors") and self.sprite_scaling_factors is not None:
                image = image / self.sprite_scaling_factors.view(1, -1, 1, 1)

            
            if self.using_block_embeds:
                """Code copied over from level_dataset, should give limited support for block embeddings"""
                # Reshape sample to [batch_size * height * width, embedding_dim]
                batch_size, embedding_dim, height, width = image.shape
                
                flat_samples = image.permute(0, 2, 3, 1).reshape(-1, embedding_dim)
                
                # Normalize vectors for cosine similarity
                flat_samples = F.normalize(flat_samples, p=2, dim=1).cpu()
                block_embeddings = F.normalize(self.block_embeddings, p=2, dim=1)

                # Calculate cosine similarity between each position and all tile embeddings
                similarities = torch.matmul(flat_samples, block_embeddings.t())
                
                # Get indices of most similar tiles
                indices = torch.softmax(similarities, dim=1)
                
                
                # Reshape back to [batch_size, height, width]
                indices = indices.reshape(batch_size, height, width, 13)
                indices = indices.permute(0, 3, 1, 2)

                image=indices.detach().cpu()
            else:
                image = F.softmax(image, dim=1)
                image = image.detach().cpu() 

            if not return_dict:
                return (image,)

            return ImagePipelineOutput(images=image)

    def print_unet_architecture(self):
        """Prints the architecture of the UNet model."""
        print(self.unet)