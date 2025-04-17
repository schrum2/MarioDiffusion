"""
Represents a bunch of configuration settings for a call
to Stable Diffusion. Can be mutated to change the configuration.
"""

import random
import torch

MUTATE_MAX_STEP_DELTA = 10
MUTATE_MAX_GUIDANCE_DELTA = 1.0

SEED_CHANGE_RATE = 0.1
USE_PROMPT_EMBEDDING_MUTATION_RATE = 0.5
USE_INITIAL_LATENTS_MUTATION_RATE = 0.5
LATENT_NOISE_SCALE = 0.1

genome_id = 0

def display_embeddings(embeds):
    if embeds == None:
        return "None"
    else:
        return "Numeric Embeddings"

def perturb_latents(latents):
    return latents + LATENT_NOISE_SCALE * torch.randn_like(latents)

class SDGenome:
    def __init__(self, model, text_embedding_capture, size, prompt, negative_prompt, seed, steps, guidance_scale, randomize = True, parent_id = None, prompt_embeds = None, negative_prompt_embeds = None, pooled_prompt_embeds = None, negative_pooled_prompt_embeds = None, strength = 0.0, latents = None):
        self.model = model
        self.text_embedding_capture = text_embedding_capture
        self.size = size
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.num_inference_steps = steps
        self.guidance_scale = guidance_scale
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
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
            f"SDGenome(model={self.model},\n"
            f"size={self.size},\n"
            f"id={self.id},\n"
            f"parent_id={self.parent_id},\n"
            f"prompt=\"{self.prompt}\",\n"
            f"negative_prompt=\"{self.negative_prompt}\",\n"
            f"seed={self.seed},\n"
            f"steps={self.num_inference_steps},\n"
            f"guidance={self.guidance_scale},\n"
            f"prompt_embeds={display_embeddings(self.prompt_embeds)},\n"
            f"negative_prompt_embeds={display_embeddings(self.negative_prompt_embeds)},\n"
            f"pooled_prompt_embeds={display_embeddings(self.pooled_prompt_embeds)},\n"
            f"negative_pooled_prompt_embeds={display_embeddings(self.negative_pooled_prompt_embeds)},\n"
            f"strength={self.strength},\n"
            f"latents={display_embeddings(self.latents)})"
        )
    
    def metadata(self):
        return {
            "model" : self.model,
            "size" : self.size,
            "id" : self.id,
            "parent_id" : self.parent_id,
            "prompt" : self.prompt,
            "negative_prompt" : self.negative_prompt,
            "seed" : self.seed,
            "num_inference_steps" : self.num_inference_steps,
            "guidance_scale" : self.guidance_scale,
            "prompt_embeds" : self.prompt_embeds,
            "negative_prompt_embeds" : self.negative_prompt_embeds,
            "pooled_prompt_embeds" : self.pooled_prompt_embeds,
            "negative_pooled_prompt_embeds" : self.negative_pooled_prompt_embeds,
            "strength" : self.strength,
            "latents" : self.latents
        }

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

    def store_text_embeddings_in_genome(self):
        if self.prompt_embeds == None:
            self.text_embedding_capture(self)

    def mutate(self):
        if random.random() < USE_INITIAL_LATENTS_MUTATION_RATE:
            if self.latents != None:
                self.latents = None
            else:
                self.store_latents_in_genome()

        if random.random() < USE_PROMPT_EMBEDDING_MUTATION_RATE:
            if self.prompt_embeds != None:
                self.prompt_embeds = None
                self.negative_prompt_embeds = None
                self.pooled_prompt_embeds = None
                self.negative_pooled_prompt_embeds = None
            else:
                self.text_embedding_capture(self)

        if random.random() < SEED_CHANGE_RATE:
            # will be a big change
            self.set_seed(random.getrandbits(64))
        else:
            # Should be a small change
            self.change_inference_steps(random.randint(-MUTATE_MAX_STEP_DELTA, MUTATE_MAX_STEP_DELTA))
            self.change_guidance_scale(random.uniform(-MUTATE_MAX_GUIDANCE_DELTA, MUTATE_MAX_GUIDANCE_DELTA))
            # Embeddings have been derived
            if isinstance(self.prompt_embeds, torch.Tensor):
                self.prompt_embeds = perturb_latents(self.prompt_embeds)
                self.negative_prompt_embeds = perturb_latents(self.negative_prompt_embeds)
            if isinstance(self.latents, torch.Tensor):
                self.latents = perturb_latents(self.latents)
                
    def mutated_child(self):
        child = SDGenome(
            self.model,
            self.text_embedding_capture,
            self.size,
            self.prompt,
            self.negative_prompt,
            self.seed,
            self.num_inference_steps,
            self.guidance_scale,
            False,
            self.id,
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
            self.strength,
            self.latents
        )
        child.mutate()
        return child
