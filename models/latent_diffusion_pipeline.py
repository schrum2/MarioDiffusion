from diffusers import DDPMPipeline
import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.ddpm.pipeline_ddpm import ImagePipelineOutput
import util.common_settings as common_settings
import os
import json
from models.general_training_helper import get_scene_from_embeddings

class UnconditionalDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, block_embeddings=None):
        super().__init__(unet, scheduler)

        self.block_embeddings = block_embeddings
    

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        super().save_pretrained(save_directory)
        # Save block_embeddings tensor if it exists
        if self.block_embeddings is not None:
            torch.save(self.block_embeddings, os.path.join(save_directory, "block_embeddings.pt"))

    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        pipeline = super().from_pretrained(pretrained_model_path, **kwargs)
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

            
            if self.block_embeddings is not None:
                image = get_scene_from_embeddings(image, self.block_embeddings)
            else:
                image = F.softmax(image, dim=1)
                image = image.detach().cpu() 

            if not return_dict:
                return (image,)

            return ImagePipelineOutput(images=image)

    def print_unet_architecture(self):
        """Prints the architecture of the UNet model."""
        print(self.unet)