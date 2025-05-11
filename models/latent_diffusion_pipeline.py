from diffusers import DDPMPipeline
import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.ddpm.pipeline_ddpm import ImagePipelineOutput

class UnconditionalDDPMPipeline(DDPMPipeline):
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        height: int = 16, width: int = 16, 
        latents: Optional[torch.FloatTensor] = None,
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

            for t in self.progress_bar(self.scheduler.timesteps):
                #print(image.shape)
                model_output = self.unet(image, t).sample
                image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

            # Why is this code not in the conditional model?
            image = F.softmax(image, dim=1)
            image = image.detach().cpu() 

            if not return_dict:
                return (image,)

            return ImagePipelineOutput(images=image)

    def print_unet_architecture(self):
        """Prints the architecture of the UNet model."""
        print(self.unet)