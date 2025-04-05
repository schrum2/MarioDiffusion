from interactive_generation import InteractiveGeneration
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler, DDPMPipeline
from level_dataset import visualize_samples
import random
from PIL import Image
import matplotlib.colors as mcolors
from tokenizer import Tokenizer 
from models import TransformerModel
from text_diffusion_pipeline import TextConditionalDDPMPipeline
from level_dataset import visualize_samples
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate levels using a trained diffusion model")
    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    # parser.add_argument("--save_as_json", action="store_true", help="Save generated levels as JSON")
    
    # Text conditional model    
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--mlm_model_file", type=str, default=os.path.join("mlm","mlm_transformer.pth"), help="Path to pre-trained text embedding model")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Text embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for text model")

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

        self.tokenizer = Tokenizer()
        self.tokenizer.load(args.pkl)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_size = self.tokenizer.get_vocab_size()
        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim
        self.text_encoder = TransformerModel(vocab_size, embedding_dim, hidden_dim).to(self.device)
        self.text_encoder.load_state_dict(torch.load(args.mlm_model_file, map_location=self.device))
        self.text_encoder.eval()  # Set to evaluation mode
        # TODO: Add support for loading from pretrained
        self.pipe = TextConditionalDDPMPipeline(
            unet=UNet2DConditionModel.from_pretrained(os.path.join(args.model_path, "unet")),
            scheduler=DDPMScheduler.from_pretrained(os.path.join(args.model_path, "scheduler")),
            text_encoder=self.text_encoder
        ).to(self.device)

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
        sample_caption_tokens = self.tokenizer.encode_batch(sample_captions)
        sample_caption_tokens = torch.tensor(sample_caption_tokens).to(self.device)

        param_values["captions"] = sample_caption_tokens

        return dict() # nothing extra here

if __name__ == "__main__":
    args = parse_args()
    ig = InteractiveLevelGeneration(args)
    ig.start()

