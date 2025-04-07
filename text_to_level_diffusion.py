from interactive_generation import InteractiveGeneration
import torch
from level_dataset import visualize_samples
from text_diffusion_pipeline import TextConditionalDDPMPipeline
from level_dataset import visualize_samples
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate levels using a trained diffusion model")    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")

    return parser.parse_args()

class InteractiveLevelGeneration(InteractiveGeneration):

    def __init__(self, args):
        InteractiveGeneration.__init__(self, {
            "prompt" : str,
            # "negative_prompt" : str,
            "start_seed" : int,
            "end_seed" : int,
            "num_inference_steps" : int
        })

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(self.device)
        self.pipe = TextConditionalDDPMPipeline.from_pretrained(args.model_path).to(self.device)
        #print(next(self.pipe.text_encoder.parameters()).device)

    def generate_image(self, param_values, generator, **extra_params):
        images = self.pipe(
            generator=generator,
            **param_values
        ).images

        return visualize_samples(images)

    def get_extra_params(self, param_values): 
        # param_values["guidance_scale"] = 8.5
        param_values["batch_size"] = 1
        param_values["output_type"] = "tensor" 

        prompt = param_values["prompt"]
        del param_values["prompt"]

        sample_captions = [prompt] # batch of size 1
        sample_caption_tokens = self.pipe.text_encoder.tokenizer.encode_batch(sample_captions)
        sample_caption_tokens = torch.tensor(sample_caption_tokens).to(self.device)

        param_values["captions"] = sample_caption_tokens

        return dict() # nothing extra here

if __name__ == "__main__":
    args = parse_args()
    ig = InteractiveLevelGeneration(args)
    ig.start()

