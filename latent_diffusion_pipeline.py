from diffusers import DDPMPipeline
import torch
from typing import Optional, Union, List, Tuple
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.ddpm.pipeline_ddpm import ImagePipelineOutput

class UnconditionalDDPMPipeline(DDPMPipeline):
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        latents: Optional[torch.FloatTensor] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        if latents is not None:
            image = latents.to(self.device)
        else:
            if isinstance(self.unet.config.sample_size, int):
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels,
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )
            else:
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels,
                    *self.unet.config.sample_size,
                )

            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet(image, t).sample
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
