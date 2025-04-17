"""
Latent noise for diffusion model input. Can be mutated to change the configuration.
"""

import random
import torch

MUTATE_MAX_STEP_DELTA = 10
MUTATE_MAX_GUIDANCE_DELTA = 1.0

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

class DiffusionGenome:
    def __init__(self, size, seed, steps, guidance_scale, randomize = True, parent_id = None, strength = 0.0, latents = None):
        self.size = size
        self.seed = seed
        self.num_inference_steps = steps
        self.guidance_scale = guidance_scale
        self.strength = strength
        self.latents = latents
        
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
            f"DiffusionGenome(size={self.size},\n"
            f"id={self.id},\n"
            f"parent_id={self.parent_id},\n"
            f"seed={self.seed},\n"
            f"steps={self.num_inference_steps},\n"
            f"guidance={self.guidance_scale},\n"
            f"strength={self.strength},\n"
            f"latents={display_embeddings(self.latents)})"
        )
    
    def metadata(self):
        return {
            "size" : self.size,
            "id" : self.id,
            "parent_id" : self.parent_id,
            "seed" : self.seed,
            "num_inference_steps" : self.num_inference_steps,
            "guidance_scale" : self.guidance_scale,
            "strength" : self.strength,
            "latents" : self.latents
        }

    # DELETE THIS?
    def store_latents_in_genome(self):
        if self.latents == None:
            # Create the initial noise latents (this is what the pipeline does internally)
            height = self.size
            width = self.size
            num_channels_latents = 4 # Always 4? # self.pipe.unet.config.in_channels
            latents_shape = (1, num_channels_latents, height // 8, width // 8)
            self.latents = torch.randn(
                latents_shape, 
                generator=torch.Generator("cuda").manual_seed(self.seed), 
                device="cuda", 
                dtype=torch.float16
            ).to("cpu")

    def mutate(self):
        if random.random() < SEED_CHANGE_RATE:
            # will be a big change
            self.set_seed(random.getrandbits(64))
        else:
            # Should be a small change
            self.change_inference_steps(random.randint(-MUTATE_MAX_STEP_DELTA, MUTATE_MAX_STEP_DELTA))
            self.change_guidance_scale(random.uniform(-MUTATE_MAX_GUIDANCE_DELTA, MUTATE_MAX_GUIDANCE_DELTA))
            self.latents = perturb_latents(self.latents)
                
    def mutated_child(self):
        child = DiffusionGenome(
            self.size,
            self.seed,
            self.num_inference_steps,
            self.guidance_scale,
            False,
            self.id,
            self.strength,
            self.latents
        )
        child.mutate()
        return child
