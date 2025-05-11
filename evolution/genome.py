"""
Latent noise for diffusion model input. Can be mutated to change the configuration.
"""

import random
import torch

MUTATE_MAX_STEP_DELTA = 10
MUTATE_MAX_GUIDANCE_DELTA = 1.0
MUTATE_MAX_SEGMENTS_DELTA = 1
MUTATE_MAX_WIDTH_DELTA = 3

SEED_CHANGE_RATE = 0.1
LATENT_NOISE_SCALE = 0.1

genome_id = 0

def display_embeddings(embeds):
    if embeds == None:
        return "None"
    else:
        return "Numeric Embeddings"

def perturb_latents(latents):
    return latents + LATENT_NOISE_SCALE * torch.randn_like(latents)

class LatentGenome:
    def __init__(self, width, seed, steps, guidance_scale, randomize = True, parent_id = None, strength = 0.0, latents = None, scene = None, prompt = None, caption = None, num_segments = 1):
        self.num_segments = num_segments
        self.width = width
        self.seed = seed
        self.num_inference_steps = steps
        self.guidance_scale = guidance_scale
        self.strength = strength
        self.latents = latents
        self.scene = scene
        self.prompt = prompt
        self.caption = caption
        
        if randomize: 
            # Randomize all aspects of picture. Seed will drastically change it
            self.set_seed(random.getrandbits(64))
            self.change_inference_steps(random.randint(-MUTATE_MAX_STEP_DELTA, MUTATE_MAX_STEP_DELTA))
            self.change_guidance_scale(random.uniform(-MUTATE_MAX_GUIDANCE_DELTA, MUTATE_MAX_GUIDANCE_DELTA))
        
        global genome_id
        self.id = genome_id
        genome_id += 1
        self.parent_id = parent_id
        self.image = None

    def set_image(self, image):
        """ save phenotype so code does not have to regenerate """
        self.image = image

    def set_seed(self, new_seed):
        self.seed = new_seed 

    def change_inference_steps(self, delta):
        self.num_inference_steps += delta
        self.num_inference_steps = max(1, self.num_inference_steps) # do not go below 1 step

    def change_guidance_scale(self, delta):
        self.guidance_scale += delta
        self.guidance_scale = max(1.0, self.guidance_scale) # Do not go below 1.0

    def __str__(self):
        return (
            f"DiffusionGenome(width={self.width},\n"
            f"id={self.id},\n"
            f"parent_id={self.parent_id},\n"
            f"seed={self.seed},\n"
            f"steps={self.num_inference_steps},\n"
            f"guidance={self.guidance_scale},\n"
            f"strength={self.strength},\n"
            f"scene={self.scene},\n"
            f"latents={display_embeddings(self.latents)},\n"
            f"caption={self.caption},\n"
            f"prompt={self.prompt},\n"
            f"width={self.width},\n"
            f"num_segments={self.num_segments})"
        )
    
    def metadata(self):
        return {
            "width" : self.width,
            "id" : self.id,
            "parent_id" : self.parent_id,
            "seed" : self.seed,
            "num_inference_steps" : self.num_inference_steps,
            "guidance_scale" : self.guidance_scale,
            "strength" : self.strength,
            "scene" : self.scene,
            "latents" : self.latents,
            "prompt" : self.prompt,
            "caption" : self.caption,
            "num_segments" : self.num_segments
        }

    def mutate(self):
        if random.random() < SEED_CHANGE_RATE:
            # will be a big change
            self.set_seed(random.getrandbits(64))
        else:
            # Should be a small change
            self.change_inference_steps(random.randint(-MUTATE_MAX_STEP_DELTA, MUTATE_MAX_STEP_DELTA))
            self.change_guidance_scale(random.uniform(-MUTATE_MAX_GUIDANCE_DELTA, MUTATE_MAX_GUIDANCE_DELTA))
            self.change_segments(random.randint(-MUTATE_MAX_SEGMENTS_DELTA, MUTATE_MAX_SEGMENTS_DELTA))
            self.change_width(random.randint(-MUTATE_MAX_WIDTH_DELTA, MUTATE_MAX_WIDTH_DELTA))
            self.latents = perturb_latents(self.latents)
            
    def change_width(self, delta):
        """Change the width of the genome and adjust latents accordingly."""
        new_width = self.width + delta
        # A width divisible by 4 is required by the unconditional model
        # because it has two downsampling layers with a stride of 2.
        # At least, this is the default configuration. Different architectures
        # would result in different requirements.
        # Long-term, I should consider fixing the unconditional pipeline
        # by sufficiently padding the inputs and cropping out the excess
        # at the end.
        new_width = (new_width + 3) // 4 * 4  # Round up to the nearest number divisible by 4

        # Clip new_width to the range [16, 64]
        new_width = max(16, min(new_width, 64))

        # Adjust latents to match the new width
        if self.latents is not None:
            _, num_channels, height, current_width = self.latents.shape
            if new_width < current_width:
                # Chop off excess width
                self.latents = self.latents[:, :, :, :new_width]
            elif new_width > current_width:
                # Expand width with random values
                additional_width = new_width - current_width
                random_values = LATENT_NOISE_SCALE * torch.randn((1, num_channels, height, additional_width), device=self.latents.device)
                self.latents = torch.cat((self.latents, random_values), dim=3)

            # Perturb latents after resizing
            self.latents = perturb_latents(self.latents)

        self.width = new_width

    def change_segments(self, delta):
        self.num_segments += delta
        self.num_segments = max(1, self.num_segments)

    def mutated_child(self):
        child = LatentGenome(
            self.width,
            self.seed,
            self.num_inference_steps,
            self.guidance_scale,
            False,
            self.id,
            self.strength,
            self.latents,
            self.scene,
            self.prompt,
            self.caption,
            self.num_segments
        )
        child.mutate()
        return child
