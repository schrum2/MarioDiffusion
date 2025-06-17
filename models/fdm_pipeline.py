#Save, Load, init, call (produce output from input)

import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional
import os
import json
# Running the main at the end of this requires messing with this import
from models.text_model import TransformerModel  
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import util.common_settings as common_settings
from safetensors.torch import save_file, load_file
from models.fdm import Gen
import models.sentence_transformers_helper as st_helper


class PipelineOutput(NamedTuple):
    images: torch.Tensor



class FDMPipeline():
    def __init__(self, tokenizer, text_encoder, model, device):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.model = model
        self.model_name = model.model_name
        self.embedding_dim = model.embedding_dim
        self.z_dim = model.z_dim
        self.kern_size = model.kern_size
        self.filter_count = model.filter_count
        self.num_res_blocks = model.num_res_blocks
        self.out_channels = model.out_channels
        self.device = device


    def to(self, device):
        self.text_encoder.to(device)
        self.model.to(device)
        self.device = device
        return self
    

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        # Save model weights
        save_file(self.model.state_dict(), os.path.join(save_directory, "model.safetensors"))

        # Save model config
        config = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "z_dim": self.z_dim,
            "kern_size": self.kern_size,
            "filter_count": self.filter_count,
            "num_res_blocks": self.num_res_blocks,
            "out_channels": self.out_channels,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)


        # Save custom text encoder
        if self.text_encoder is not None:
            self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        if self.tokenizer is not None and hasattr(self.tokenizer, 'save_pretrained'):
            # Save tokenizer if it has a save_pretrained method.
            # Otherwise, we presume the tokenizer was saved by the text encoder.
            self.tokenizer.save_pretrained(os.path.join(save_directory, "text_encoder"))
            
    


    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):

        tokenizer = None
        text_encoder_path = os.path.join(pretrained_model_path, "text_encoder")
        if os.path.exists(text_encoder_path): #Should always be a pretrained model
            text_encoder = AutoModel.from_pretrained(text_encoder_path, local_files_only=True, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, local_files_only=True)
        else:
            text_encoder = None


        # Load the model
        with open(os.path.join(pretrained_model_path, "config.json")) as f:
            config = json.load(f)

        model = Gen(**config)

        

        # Load weights
        print(f"Loading model from {os.path.join(pretrained_model_path, 'model.safetensors')}")
        state_dict = load_file(os.path.join(pretrained_model_path, "model.safetensors"))


        model.load_state_dict(state_dict)


        # Instantiate your pipeline
        pipeline = cls(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            model=model,
            device=None,
            **kwargs,
        )
        return pipeline
    

    def __call__(
            self,
            caption: Optional[str | list[str]] = None,
            generator: Optional[torch.Generator] = None,
            batch_size: int = 1,
            show_progress_bar: bool = True,
            noise_vector: Optional[torch.FloatTensor] = None,
    ):
        """Generate a batch of images based on text input using the five-dollar-model.

        Args:
            caption: Text description(s) of the desired output. Can be a string or list of strings.
            generator: Random number generator for reproducibility.
            batch_size: Number of samples to generate in parallel.
            show_progress_bar: Whether to show a progress bar during generation.
            noise_vector: Optional noise vector for the generation process. Must match the expected length of the noise vector.

        Returns:
            PipelineOutput containing the generated image tensor (batch_size, ...).
        """

        if caption is not None and self.text_encoder is None:
            raise ValueError("Text encoder is required for conditional generation")
        
        if self.text_encoder is not None:
            self.text_encoder.to(self.device)
            self.text_encoder.eval()
        
        with torch.no_grad():
            captions = self._prepare_text_batch(caption, batch_size, "caption")
           # --- Prepare text embeddings ---
            if captions is not None:
                text_embeddings = st_helper.encode(captions, self.tokenizer, self.text_encoder, self.device)
            else:
                # Unconditional generation: use unconditional embeddings only
                text_embeddings = st_helper.encode([""] * batch_size, self.tokenizer, self.text_encoder, self.device)           
            text_embeddings = text_embeddings*6 #Multiply by a scaling factor, this helps prevent errors later
            
            if noise_vector is not None:
                outputs = self.model(text_embeddings, noise_vector)
            else:
                noiseVec = torch.randn(text_embeddings.shape[0], self.model.z_dim, device=self.device, generator=generator)
                outputs = self.model(text_embeddings, noiseVec)

            outputs = F.softmax(outputs, dim=1)


        return PipelineOutput(images=outputs)



        # --- Handle batching for captions ---
    def _prepare_text_batch(self, text: Optional[str | list[str]], batch_size: int, name: str) -> Optional[list[str]]:
        if text is None:
            return None
        if isinstance(text, str):
            return [text] * batch_size
        if isinstance(text, list):
            if len(text) == 1:
                return text * batch_size
            if len(text) != batch_size:
                raise ValueError(f"{name} list length {len(text)} does not match batch_size {batch_size}")
            return text
        raise ValueError(f"{name} must be a string or list of strings")


    def _encode_token_captions(self, captions, max_length):
        """
        Helper method to encode and pad captions to fixed length.
        This approach specifically applies to a text encoder that
        creates token embeddings, like my TransformerModel
        """
        caption_ids = []
        for cap in captions:
            ids = self.tokenizer.encode(cap)
            ids = torch.tensor(ids, device=self.device)
            if ids.shape[0] > max_length:
                raise ValueError(f"Caption length {ids.shape[0]} exceeds max sequence length of {max_length}")
            elif ids.shape[0] < max_length:
                padding = torch.zeros(max_length - ids.shape[0], dtype=ids.dtype, device=self.device)
                ids = torch.cat([ids, padding], dim=0)
            caption_ids.append(ids.unsqueeze(0))
        return torch.cat(caption_ids, dim=0)



if __name__ == "__main__":

    import os
    import torch
    from level_dataset import visualize_samples


    # Example usage
    model_path = "SMB1-conditional-fdm-test"
    pipe = FDMPipeline.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)

    output = pipe("A beautiful sunset over the mountains", batch_size=4)
    

    sample_images = visualize_samples(output.images, use_tiles=True)
    sample_images.show()