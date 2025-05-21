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
            
class PipelineOutput(NamedTuple):
    images: torch.Tensor

# Create a custom pipeline for text-conditional generation
class TextConditionalDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, text_encoder=None, tokenizer=None):
        super().__init__(unet=unet, scheduler=scheduler)
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.supports_negative_prompt = hasattr(unet, 'negative_prompt_support') and unet.negative_prompt_support

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

        # Save custom text encoder
        if self.text_encoder is not None:
            self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        if self.tokenizer is not None and hasattr(self.tokenizer, 'save_pretrained'):
            # Save tokenizer if it has a save_pretrained method.
            # Otherwise, we presume the tokenizer was saved by the text encoder.
            self.tokenizer.save_pretrained(os.path.join(save_directory, "text_encoder"))
            
        # Save supports_negative_prompt flag
        with open(os.path.join(save_directory, "pipeline_config.json"), "w") as f:
            json.dump({
                "supports_negative_prompt": self.supports_negative_prompt,
                "text_encoder_type": type(self.text_encoder).__name__   
            }, f)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, using_pretrained = False, **kwargs):
        #from diffusers.utils import load_config, load_state_dict
        # Load model_index.json
        #model_index = load_config(pretrained_model_path)

        # Load components manually
        unet_path = os.path.join(pretrained_model_path, "unet")
        unet = UNet2DConditionModel.from_pretrained(unet_path)

        scheduler_path = os.path.join(pretrained_model_path, "scheduler")
        # Have heard that DDIMScheduler might be faster for inference, though not necessarily better
        scheduler = DDPMScheduler.from_pretrained(scheduler_path)

        tokenizer = None
        text_encoder_path = os.path.join(pretrained_model_path, "text_encoder")
        if os.path.exists(text_encoder_path):
            try:
                text_encoder = AutoModel.from_pretrained(text_encoder_path, local_files_only=True)
                tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, local_files_only=True)
            except (ValueError, KeyError):
                text_encoder = TransformerModel.from_pretrained(text_encoder_path)
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
        # Load supports_negative_prompt flag if present
        config_path = os.path.join(pretrained_model_path, "pipeline_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            pipeline.supports_negative_prompt = config.get("supports_negative_prompt", False)
        return pipeline
        
    def __call__(
        self,
        caption: Optional[str | list[str]] = None,
        negative_prompt: Optional[str | list[str]] = None,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = common_settings.NUM_INFERENCE_STEPS,
        guidance_scale: float = common_settings.GUIDANCE_SCALE,
        height: int = 16,
        width: int = 16,
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
        # Validate text encoder if we need it
        if caption is not None and self.text_encoder is None:
            raise ValueError("Text encoder is required for conditional generation")

        self.unet.eval()
        if self.text_encoder is not None:
            self.text_encoder.to(self.device)
            self.text_encoder.eval()

        with torch.no_grad():
            # --- Handle batching for captions ---
            def prepare_text_batch(text: Optional[str | list[str]], batch_size: int, name: str) -> Optional[list[str]]:
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

            captions = prepare_text_batch(caption, batch_size, "caption")
            negatives = prepare_text_batch(negative_prompt, batch_size, "negative_prompt")

            # --- Prepare text embeddings ---
            if(isinstance(self.text_encoder, TransformerModel)):
                if captions is not None:
                    max_length = self.text_encoder.max_seq_length
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
                    caption_ids = torch.cat(caption_ids, dim=0)  # (batch_size, max_length)
                    caption_embedding = self.text_encoder.get_embeddings(caption_ids)

                    # Handle negative prompt if provided
                    if negatives is not None:
                        if not self.supports_negative_prompt:
                            raise ValueError("This model was not trained with negative prompt support")
                        negative_ids = []
                        for neg in negatives:
                            ids = self.tokenizer.encode(neg)
                            ids = torch.tensor(ids, device=self.device)
                            if ids.shape[0] > max_length:
                                raise ValueError(f"Negative caption length {ids.shape[0]} exceeds max sequence length of {max_length}")
                            elif ids.shape[0] < max_length:
                                padding = torch.zeros(max_length - ids.shape[0], dtype=ids.dtype, device=self.device)
                                ids = torch.cat([ids, padding], dim=0)
                            negative_ids.append(ids.unsqueeze(0))
                        negative_ids = torch.cat(negative_ids, dim=0)
                        negative_embedding = self.text_encoder.get_embeddings(negative_ids)

                        # Get unconditional (empty) embedding
                        empty_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=self.device)
                        empty_embedding = self.text_encoder.get_embeddings(empty_ids)

                        # Concatenate [negative, unconditional, conditional] along batch
                        text_embeddings = torch.cat([negative_embedding, empty_embedding, caption_embedding], dim=0)
                    else:
                        # Standard classifier-free guidance with just [unconditional, conditional]
                        empty_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=self.device)
                        empty_embedding = self.text_encoder.get_embeddings(empty_ids)
                        text_embeddings = torch.cat([empty_embedding, caption_embedding], dim=0)
                else:
                    # For unconditional generation, use empty embeddings matching max_seq_length
                    max_length = self.text_encoder.max_seq_length
                    empty_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=self.device)
                    text_embeddings = self.text_encoder.get_embeddings(empty_ids)
            else: #Case for the pre-trained text encoder
                def mean_pooling(model_output, attention_mask):
                    token_embeddings = model_output.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                #Encode text
                def encode(texts):
                    # Tokenize sentences
                    encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
                    encoded_input = encoded_input.to(self.device)
                    # Compute token embeddings
                    with torch.no_grad():
                        model_output = self.text_encoder(**encoded_input, return_dict=True)

                    # Perform pooling
                    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

                    # Normalize embeddings
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    return embeddings
                if captions is not None:

                    text_embeddings = encode(captions)
                    uncond_embeddings = encode([""] * batch_size)

                    if negatives is not None:
                        # Negative prompt embeddings
                        neg_tokens = self.tokenizer(negatives, return_tensors="pt", padding=True, truncation=True).to(self.device)
                        neg_embeddings = self.text_encoder(**neg_tokens).last_hidden_state  # [batch, seq_len, hidden_size]
                        # Concatenate [neg, uncond, cond]
                        text_embeddings = torch.cat([neg_embeddings, uncond_embeddings, text_embeddings], dim=0)
                    else:
                        # Concatenate [uncond, cond]
                        
                        text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

                else:
                    # Unconditional generation: use unconditional embeddings only
                    text_embeddings = encode([""] * batch_size)
                text_embeddings = text_embeddings.unsqueeze(1)  # (batch_size, 1, hidden_size)
            
            



            # --- Set up initial latent state ---
            device = self.device
            sample_shape = (batch_size, self.unet.config.in_channels, height, width)

            if raw_latent_sample is not None:
                if input_scene is not None:
                    raise ValueError("Cannot provide both raw_latent_sample and input_scene")
                sample = raw_latent_sample.to(device)
                if sample.shape[1] != sample_shape[1]:
                    raise ValueError(f"Wrong number of channels in raw_latent_sample: Expected {self.unet.config.in_channels} but got {sample.shape[1]}")
                if sample.shape[0] == 1 and batch_size > 1:
                    sample = sample.repeat(batch_size, 1, 1, 1)
                elif sample.shape[0] != batch_size:
                    raise ValueError(f"raw_latent_sample batch size {sample.shape[0]} does not match batch_size {batch_size}")
            elif input_scene is not None:
                # input_scene can be (H, W) or (batch_size, H, W)
                scene_tensor = torch.tensor(input_scene, dtype=torch.long, device=device)
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
                sample = torch.randn(sample_shape, generator=generator, device=device)

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
                # Apply softmax to get probabilities for each tile type
                sample = F.softmax(sample, dim=1)
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

if __name__ == "__main__":
    import os
    import torch
    from level_dataset import visualize_samples

    # This won't run unless some imports at the top of the file are modified

    # Set up the pipeline
    model_path = "prev-cond-model"
    pipe = TextConditionalDDPMPipeline.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)

    # Generate with test prompt
    output = pipe(
        caption="full floor. two pipes.",
        num_inference_steps=50,
        guidance_scale=7.5,
        height=16,
        width=16,
    )

    # Convert output to proper format for visualization
    sample_images = visualize_samples(output.images, use_tiles=True)
    sample_images.show()

    # The visualize_samples function will save the image and also return the tile indices
    print("Generation complete! Check the generated image.")
