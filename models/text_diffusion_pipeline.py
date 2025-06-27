import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional
import os
from diffusers import DDPMPipeline, UNet2DConditionModel, DDPMScheduler
import json
# Running the main at the end of this requires messing with this import
from models.text_model import TransformerModel  
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import util.common_settings as common_settings
import models.sentence_transformers_helper as st_helper
import models.text_model as text_model
from models.general_training_helper import get_scene_from_embeddings
            
class PipelineOutput(NamedTuple):
    images: torch.Tensor
    


# Create a custom pipeline for text-conditional generation
class TextConditionalDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, text_encoder=None, tokenizer=None, supports_pretrained_split=False, block_embeddings=None):
        super().__init__(unet=unet, scheduler=scheduler)
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.supports_negative_prompt = hasattr(unet, 'negative_prompt_support') and unet.negative_prompt_support
        self.supports_pretrained_split = supports_pretrained_split
        self.block_embeddings = block_embeddings

        if self.tokenizer is None and self.text_encoder is not None:
            # Use the tokenizer from the text encoder if not provided
            self.tokenizer = self.text_encoder.tokenizer
        
        # Register the text_encoder so that .to(), .cpu(), .cuda(), etc. work correctly
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
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

        # Save block_embeddings tensor if it exists
        if self.block_embeddings is not None:
            torch.save(self.block_embeddings, os.path.join(save_directory, "block_embeddings.pt"))

        # Save supports_negative_prompt and supports_pretrained_split flags
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            json.dump({
                "supports_negative_prompt": self.supports_negative_prompt,
                "supports_pretrained_split": self.supports_pretrained_split,
                "text_encoder_type": type(self.text_encoder).__name__   
            }, f)


        #Text encoder/tokenizer saving is different depending on if we're using a larger pretrained model
        if isinstance(self.text_encoder, TransformerModel):
            # Save custom text encoder
            if self.text_encoder is not None:
                self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        else:
            #Save pretrained tokenizer by name, so we can load from huggingface instead of saving a giant local model
            text_encoder_info = {
                "text_encoder_name": self.text_encoder.config.name_or_path,
                "tokenizer_name": self.tokenizer.name_or_path,
            }

            text_encoder_directory = os.path.join(save_directory, "text_encoder")
            os.makedirs(text_encoder_directory, exist_ok=True)

            with open(os.path.join(text_encoder_directory, "loading_info.json"), "w") as f:
                json.dump(text_encoder_info, f)
            


    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        # Load UNet - can be local path or HF model ID with subfolder
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
        scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

        tokenizer = None
        text_encoder_path = os.path.join(pretrained_model_path, "text_encoder")
        
        if os.path.exists(text_encoder_path):
            # Test for the new saving system, where we save a simple config file
            # This case is for pretrained text encoders: MiniLM, GTE
            if os.path.exists(os.path.join(text_encoder_path, "loading_info.json")):
                with open(os.path.join(text_encoder_path, "loading_info.json"), "r") as f:
                    encoder_config = json.load(f)

                text_encoder = AutoModel.from_pretrained(encoder_config['text_encoder_name'], trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(encoder_config['tokenizer_name'])
            
            #Legacy loading system, loads models directly if the whole thing is saved in the directory
            else:
                try: # Assumes MiniLM or GTE is directly saved on the disk in subdir
                    text_encoder = AutoModel.from_pretrained(text_encoder_path, local_files_only=True, trust_remote_code=True)
                    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, local_files_only=True)
                except (ValueError, KeyError):
                    # If it fails, assume it's a custom TransformerModel 
                    # This command can grab code from local directory or Hugging Face
                    text_encoder = TransformerModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
                    tokenizer = text_encoder.tokenizer
        else:
            text_encoder = None

        # Instantiate your pipeline
        pipeline = cls(
            unet=unet,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            **kwargs,
        )

        #Loads block embeddings if present
        block_embeds_path = os.path.join(pretrained_model_path, "block_embeddings.pt")
        if os.path.exists(block_embeds_path):
            pipeline.block_embeddings = torch.load(block_embeds_path, map_location="cpu")
        else:
            pipeline.block_embeddings = None
        

        # Load supports_negative_prompt flag if present
        config_path = os.path.join(pretrained_model_path, "pipeline_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            pipeline.supports_negative_prompt = config.get("supports_negative_prompt", False)
            pipeline.supports_pretrained_split = config.get("supports_pretrained_split", False)
        return pipeline

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

    def _prepare_initial_sample(self, 
                                raw_latent_sample: Optional[torch.Tensor],
                                input_scene: Optional[torch.Tensor],
                                batch_size: int, height: int, width: int,
                                generator: Optional[torch.Generator]) -> torch.Tensor:
        """Prepare the initial sample for diffusion."""
 
        sample_shape = (batch_size, self.unet.config.in_channels, height, width)

        if raw_latent_sample is not None:
            if input_scene is not None:
                raise ValueError("Cannot provide both raw_latent_sample and input_scene")
            sample = raw_latent_sample.to(self.device)
            if sample.shape[1] != sample_shape[1]:
                raise ValueError(f"Wrong number of channels in raw_latent_sample: Expected {self.unet.config.in_channels} but got {sample.shape[1]}")
            if sample.shape[0] == 1 and batch_size > 1:
                sample = sample.repeat(batch_size, 1, 1, 1)
            elif sample.shape[0] != batch_size:
                raise ValueError(f"raw_latent_sample batch size {sample.shape[0]} does not match batch_size {batch_size}")
        elif input_scene is not None:
            # input_scene can be (H, W) or (batch_size, H, W)
            scene_tensor = torch.tensor(input_scene, dtype=torch.long, device=self.device)
            if scene_tensor.dim() == 2:
                # (H, W) -> repeat for batch
                scene_tensor = scene_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
            elif scene_tensor.shape[0] == 1 and batch_size > 1:
                scene_tensor = scene_tensor.repeat(batch_size, 1, 1)
            elif scene_tensor.shape[0] != batch_size:
                raise ValueError(f"input_scene batch size {scene_tensor.shape[0]} does not match batch_size {batch_size}")
            # One-hot encode: (batch, H, W, C)
            one_hot = F.one_hot(scene_tensor, num_classes=self.unet.config.in_channels).float()
            # (batch, H, W, C) -> (batch, C, H, W)
            sample = one_hot.permute(0, 3, 1, 2)
        else:
            # Start from random noise
            sample = torch.randn(sample_shape, generator=generator, device=self.device)

        return sample

    def __call__(
        self,
        caption: Optional[str | list[str]] = None,
        negative_prompt: Optional[str | list[str]] = None,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = common_settings.NUM_INFERENCE_STEPS,
        guidance_scale: float = common_settings.GUIDANCE_SCALE,
        height: int = common_settings.MARIO_HEIGHT,
        width: int = common_settings.MARIO_WIDTH,
        raw_latent_sample: Optional[torch.FloatTensor] = None,
        input_scene: Optional[torch.Tensor] = None,
        output_type: str = "tensor",
        batch_size: int = 1,
        show_progress_bar: bool = True,
    ) -> PipelineOutput:
        """Generate a batch of images based on text input using the diffusion model.

        Args:
            caption: Text description(s) of the desired output. Can be a string or list of strings.
            negative_prompt: Text description(s) of what should not appear in the output. String or list.
            generator: Random number generator for reproducibility.
            num_inference_steps: Number of denoising steps (more = higher quality, slower).
            guidance_scale: How strongly the generation follows the text prompt (higher = stronger).
            height: Height of generated image in tiles.
            width: Width of generated image in tiles.
            raw_latent_sample: Optional starting point for diffusion instead of random noise.
                Must have correct number of channels matching the UNet.
            input_scene: Optional 2D or 3D int tensor where each value corresponds to a tile type.
                Will be converted to one-hot encoding as starting point.
            output_type: Currently only "tensor" is supported.
            batch_size: Number of samples to generate in parallel.

        Returns:
            PipelineOutput containing the generated image tensor (batch_size, ...).
        """

        #       I would like to simplify the code to this, but the AI suggestion didn't work, and 
        #       I did not feel good just pasting it all in. Will need to tackle it bit by bit.

        #        if caption is not None and self.text_encoder is None:
        #            raise ValueError("Text encoder required for conditional generation")
    
        #        self.unet.eval()
        #        if self.text_encoder is not None:
        #            self.text_encoder.to(self.device)
        #            self.text_encoder.eval()
        #
        #        with torch.no_grad():
        #            # Process text inputs
        #            captions = self.prepare_text_batch(caption, batch_size, "caption")
        #            negatives = self.prepare_text_batch(negative_prompt, batch_size, "negative_prompt")
         
        #            # Get embeddings
        #            text_embeddings = self.prepare_embeddings(captions, negatives, batch_size)
        #            
        #            # Set up initial latent state
        #            sample = self.prepare_initial_sample(raw_latent_sample, input_scene, 
        #                                              batch_size, height, width, generator)
           
        #            # Run diffusion process
        #            sample = self.run_diffusion(sample, text_embeddings, num_inference_steps,
        #                                      guidance_scale, generator, show_progress_bar,
        #                                      has_caption=caption is not None,
        #                                      has_negative=negative_prompt is not None)
         
        #            # Format output
        #            if output_type == "tensor":
        #                sample = F.softmax(sample, dim=1)
        #            else:
        #                raise ValueError(f"Unsupported output type: {output_type}")

        #        return PipelineOutput(images=sample)

        # Validate text encoder if we need it
        if caption is not None and self.text_encoder is None:
            raise ValueError("Text encoder is required for conditional generation")

        self.unet.eval()
        if self.text_encoder is not None:
            self.text_encoder.to(self.device)
            self.text_encoder.eval()

        with torch.no_grad():
            captions = self._prepare_text_batch(caption, batch_size, "caption")
            negatives = self._prepare_text_batch(negative_prompt, batch_size, "negative_prompt")

            # --- Prepare text embeddings ---
            if(isinstance(self.text_encoder, TransformerModel)):
                text_embeddings = text_model.get_embeddings(batch_size=batch_size,
                                                            tokenizer=self.text_encoder.tokenizer,
                                                            text_encoder=self.text_encoder,
                                                            captions=captions,
                                                            neg_captions=negatives,
                                                            device=self.device)
            else: #Case for the pre-trained text encoder
                if(self.supports_pretrained_split): #If we have a split flag incorporated
                    text_embeddings = st_helper.get_embeddings_split(batch_size = batch_size,
                                                            tokenizer=self.tokenizer,
                                                            model=self.text_encoder,
                                                            captions=captions,
                                                            neg_captions=negatives,
                                                            device=self.device)
                else:
                    text_embeddings = st_helper.get_embeddings(batch_size = batch_size,
                                                                tokenizer=self.tokenizer,
                                                                model=self.text_encoder,
                                                                captions=captions,
                                                                neg_captions=negatives,
                                                                device=self.device)

                            
            # --- Set up initial latent state ---
            sample = self._prepare_initial_sample(raw_latent_sample, input_scene, 
                                                 batch_size, height, width, generator)

            # --- Set up diffusion process ---
            self.scheduler.set_timesteps(num_inference_steps)

            # Denoising loop
            iterator = self.progress_bar(self.scheduler.timesteps) if show_progress_bar else self.scheduler.timesteps
            for t in iterator:
                # Handle conditional generation
                if captions is not None:
                    if negatives is not None:
                        # Three copies for negative prompt guidance
                        model_input = torch.cat([sample, sample, sample], dim=0)
                    else:
                        # Two copies for standard classifier-free guidance
                        model_input = torch.cat([sample, sample], dim=0)
                else:
                    model_input = sample

                # Predict noise residual
                model_kwargs = {"encoder_hidden_states": text_embeddings}
                noise_pred = self.unet(model_input, t, **model_kwargs).sample

                # Apply guidance
                if captions is not None:
                    if negatives is not None:
                        # Split predictions for negative, unconditional, and text-conditional
                        noise_pred_neg, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                        noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        noise_pred = noise_pred_guided - guidance_scale * (noise_pred_neg - noise_pred_uncond)
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute previous sample: x_{t-1} = scheduler(x_t, noise_pred)
                sample = self.scheduler.step(noise_pred, t, sample, generator=generator).prev_sample

            # Convert to output format
            if output_type == "tensor":
                if self.block_embeddings is not None:
                    sample = get_scene_from_embeddings(sample, self.block_embeddings)
                else:
                    # Apply softmax to get probabilities for each tile type
                    sample = F.softmax(sample, dim=1)
                    sample = sample.detach().cpu() 
            else:
                raise ValueError(f"Unsupported output type: {output_type}")

        return PipelineOutput(images=sample)

    def print_unet_architecture(self):
        """Prints the architecture of the UNet model."""
        print(self.unet)

    def print_text_encoder_architecture(self):
        """Prints the architecture of the text encoder model, if it exists."""
        if self.text_encoder is not None:
            print(self.text_encoder)
        else:
            print("No text encoder is set.")

    def save_unet_architecture_pdf(self, height, width, filename="unet_architecture", batch_size=1, device=None):
        """
        Have to separately install torchview for this to work

        Saves a visualization of the UNet architecture as a PDF using torchview.
        Args:
            height: Height of the dummy input.
            width: Width of the dummy input.
            filename: Output PDF filename.
            batch_size: Batch size for dummy input.
            device: Device to run the dummy input on (defaults to pipeline device).
        """
        from torchview import draw_graph
        import graphviz
        
        if device is None:
            device = self.device if hasattr(self, 'device') else 'cpu'
        in_channels = self.unet.config.in_channels if hasattr(self.unet, 'config') else 1
        sample_shape = tuple([batch_size, in_channels, height, width])

        dummy_x = torch.randn(size=sample_shape, device=device)
        dummy_t = torch.tensor([0] * batch_size, dtype=torch.long, device=device)

        # Prepare dummy text embedding (match what your UNet expects)
        if hasattr(self.unet, 'config') and hasattr(self.unet.config, 'cross_attention_dim'):
            cross_attention_dim = self.unet.config.cross_attention_dim
        else:
            cross_attention_dim = 128  # fallback
        encoder_hidden_states = torch.randn(batch_size, 1, cross_attention_dim, device=device)

        self.unet.eval()
        inputs = (dummy_x, dummy_t, encoder_hidden_states)
        #self.unet.down_blocks = self.unet.down_blocks[:2]

        graph = draw_graph(
            model=self.unet,
            input_data=inputs,
            expand_nested=False,
            #enable_output_shape=True,   
            #roll_out="nested",
            depth=1
        )
        #graph.visual_graph.engine = "neato"
        graph.visual_graph.attr(#rankdir="LR",
                                nodesep="0.1",      # decrease space between nodes in the same rank (default ~0.25)
                                ranksep="0.2",       # decrease space between ranks (default ~0.5)
                                concentrate="true"  # merge edges between nodes in the same rank
                            )
        graph.visual_graph.node_attr.update(
            shape="rectangle",
            width="1.5",   # narrow width
            height="0.5"  # taller height to make vertical rectangles
            #fixedsize="true"
        )
        
        graph.visual_graph.render(filename, format='pdf', cleanup=False)  # Cleanup removes intermediate files
        graph.visual_graph.save('unet_architecture.dot')

        # Save the graph to a PDF file
        print(f"UNet architecture saved to {filename}")

