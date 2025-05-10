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
    def __init__(self, width, seed, steps, guidance_scale, randomize = True, parent_id = None, strength = 0.0, latents = None, scene = None, prompt = None, caption = None, num_segments = 1, generation_width = 16):
        self.num_segments = num_segments
        self.generation_width = generation_width
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
            f"num_segments={self.num_segments},\n"
            f"generation_width={self.generation_width})"
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
            "num_segments" : self.num_segments,
            "generation_width" : self.generation_width
        }

    def mutate(self):
        if random.random() < SEED_CHANGE_RATE:
            # will be a big change
            self.set_seed(random.getrandbits(64))
        else:
            # Should be a small change
            self.change_inference_steps(random.randint(-MUTATE_MAX_STEP_DELTA, MUTATE_MAX_STEP_DELTA))
            self.change_guidance_scale(random.uniform(-MUTATE_MAX_GUIDANCE_DELTA, MUTATE_MAX_GUIDANCE_DELTA))
            self.latents = perturb_latents(self.latents)
            self.change_segments(random.randint(-MUTATE_MAX_SEGMENTS_DELTA, MUTATE_MAX_SEGMENTS_DELTA))
            self.change_width(random.randint(-MUTATE_MAX_WIDTH_DELTA, MUTATE_MAX_WIDTH_DELTA))
            
    def change_width(self, delta):
        self.generation_width += delta
        # 16 is default size of one segment. 64 is max for my GPU VRAM
        self.generation_width = max(16, self.generation_width)
        self.generation_width = min(64, self.generation_width)

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
            self.num_segments,
            self.generation_width
        )
        child.mutate()
        return child
