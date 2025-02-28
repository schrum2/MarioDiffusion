
from diffusers.utils import make_image_grid
from diffusers import EulerDiscreteScheduler

from interactive_generation import InteractiveGeneration
import os
import sys

class SDLoRAInteractiveGeneration(InteractiveGeneration):

    def __init__(self, lora_model):
        InteractiveGeneration.__init__(self, {
            "prompt" : str,
            "negative_prompt" : str,
            "start_seed" : int,
            "end_seed" : int,
            "num_inference_steps" : int
        })

        import torch
        from diffusers import StableDiffusionPipeline

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
            , custom_pipeline="lpw_stable_diffusion" # Allows token weighting, as in "A (white:1.5) cat"
            , torch_dtype = torch.float16
            , safety_checker = None
            , requires_safety_checker = False
        ).to("cuda")

        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict=lora_model
            , adapter_name = "my_lora"
            , use_safetensors=True  # Ensures it treats the file as a safetensors file
        )

        self.pipe.set_adapters(
            ["my_lora"]
            , adapter_weights = [1.0]
        )

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

    def generate_image(self, param_values, generator, **extra_params):
        image = self.pipe(
            generator=generator,
            **param_values
        ).images[0]
        return image

    def get_extra_params(self, param_values): 
        param_values["guidance_scale"] = 8.5
        param_values["height"] = 256 # For Mario levels
        param_values["width"] = 256 # For Mario levels
        return dict() # nothing extra here

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("model path needed")
        quit()

    lora_model_path = sys.argv[1]

    if not os.path.exists(lora_model_path): 
        print("Invalid model path")
        print(lora_model_path)
        quit()

    ig = SDLoRAInteractiveGeneration(lora_model_path)
    ig.start()
