from image_grid import ImageGridViewer
import tkinter as tk
import random
from genome import SDGenome
import torch
from diffusers import EulerDiscreteScheduler
from abc import ABC, abstractmethod
from models import SD_MODEL, SDXL_MODEL

class Evolver(ABC):
    def __init__(self, population_size = 9):
        self.population_size = population_size
        self.steps = 20
        self.guidance_scale = 7.5

        self.evolution_history = []
        self.generation = 0

    def get_generation(self):
        return self.generation

    def start_evolution(self):
        self.genomes = []
        self.generation = 0

        self.root = tk.Tk()
        self.viewer = ImageGridViewer(
            self.root, 
            callback_fn=self.next_generation,
            initial_prompt="",
            initial_negative_prompt="",
            back_fn=self.previous_generation,
            generation_fn=self.get_generation
        )
        # Start the GUI event loop
        self.root.mainloop()

    def previous_generation(self):
        if self.generation > 0:
            self.genomes = self.evolution_history.pop()
            self.generation -= 1
            self.fill_with_images_from_genomes()

    @abstractmethod
    def initialize_population(self):
        pass

    def next_generation(self,selected_images,prompt,negative_prompt):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        if selected_images == []:
            print("Resetting population and generations--------------------")
            self.evolution_history = []
            self.initialize_population()
            self.generation = 0
        else:
            print(f"Generation {self.generation}---------------------------")
            for (i,image) in selected_images:
                print(f"Selected for survival: {self.genomes[i]}")
                #self.genomes[i].set_image(image) # Done earlier, for ALL genomes

            # Track history of all genomes
            self.evolution_history.append(self.genomes.copy())

            # Pure elitism
            keepers = [self.genomes[i] for (i,_) in selected_images]

            children = []
            # Fill remaining slots with mutated children
            for i in range(len(keepers), self.population_size):
                g = random.choice(keepers).mutated_child() # New genome
                # prompts may have changed
                g.prompt = prompt
                g.negative_prompt = negative_prompt
                children.append(g)

            # combined population
            self.genomes = keepers + children
            self.generation += 1

        self.fill_with_images_from_genomes()

    def fill_with_images_from_genomes(self):
        self.viewer.clear_images()

        for g in self.genomes:
            
            if g.image:
                # used saved image from previous generation
                print(f"Use cached image for {g}")
                image = g.image
            else:
                image = self.generate_image(g)
                g.set_image(image)

            # Add image to viewer
            self.viewer.add_image(image, g)
        
            # Update the GUI to show new image
            self.root.update()
    
        print("Make selections and click \"Evolve\"")        

from diffusers import StableDiffusionPipeline

class SDEvolver(Evolver):
    def __init__(self):
        Evolver.__init__(self)

        print(f"Using {self.get_model()}")

        # I disabled the safety checker. There is a risk of NSFW content.
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.get_model(),
            torch_dtype=torch.float16,
            # Does not allow direct use of text embeddings
            #custom_pipeline="lpw_stable_diffusion", # Allows token weighting, as in "A (white:1.5) cat"
            safety_checker = None, # Faster
            requires_safety_checker = False
        )
        self.pipe.to("cuda")

        # Default is PNDMScheduler
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    def initialize_population(self):
        self.genomes = [SDGenome(self.get_model(), self.put_text_embeddings_in_genome, self.base_image_size(), self.prompt, self.negative_prompt, seed, self.steps, self.guidance_scale) for seed in range(self.population_size)]

    def put_text_embeddings_in_genome(self, g):
        # Need to get the embeddings for the first time
        text_embeddings, negative_text_embeddings = self.pipe.encode_prompt(
            prompt=g.prompt,
            negative_prompt=g.negative_prompt, 
            device=self.pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )

        g.prompt_embeds = text_embeddings
        g.negative_prompt_embeds = negative_text_embeddings

    def base_image_size(self):
        return 512

    def get_model(self):
        return SD_MODEL

    def generate_image(self, g):
        # generate fresh new image
        print(f"Generate new image for {g}")
        generator = torch.Generator("cuda").manual_seed(g.seed)

        settings = {
            "guidance_scale" : g.guidance_scale,
            "num_inference_steps" : g.num_inference_steps,
            "strength" : g.strength
        }

        if g.prompt_embeds != None:
            settings["prompt_embeds"] = g.prompt_embeds.to("cuda")
            settings["negative_prompt_embeds"] = g.negative_prompt_embeds.to("cuda")
            if isinstance(g.pooled_prompt_embeds, torch.Tensor):
                settings["pooled_prompt_embeds"] = g.pooled_prompt_embeds.to("cuda")
                settings["negative_pooled_prompt_embeds"] = g.negative_pooled_prompt_embeds.to("cuda")
        else:
            settings["prompt"] = g.prompt
            settings["negative_prompt"] = g.negative_prompt

        if g.latents != None:
            settings["latents"] = g.latents.to("cuda")
        
        image = self.pipe(
            generator=generator,
            **settings
        ).images[0]

        if g.prompt_embeds != None:
            g.prompt_embeds.to("cpu")
            g.negative_prompt_embeds.to("cpu")

        if g.pooled_prompt_embeds != None:
            g.pooled_prompt_embeds.to("cpu")
            g.negative_pooled_prompt_embeds.to("cpu")

        if g.latents != None:
            g.latents.to("cpu")

        return image

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)

class SDXLEvolver(SDEvolver):
    def __init__(self):
        Evolver.__init__(self, 4) # Smaller population size, generation takes so long

        print(f"Using {self.get_model()}")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.get_model(),
            torch_dtype=torch.float16
        )
        self.pipe.to("cuda")

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    def base_image_size(self):
        # Has to be larger for SDXL for the latent size calculation to work
        return 1024

    def get_model(self):
        return SDXL_MODEL
    
    def put_text_embeddings_in_genome(self, g):
        # Need to get the embeddings for the first time
        text_embeddings, negative_text_embeddings, pooled_text_embeddings, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
            prompt=g.prompt,
            negative_prompt=g.negative_prompt, 
            device=self.pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )

        g.prompt_embeds = text_embeddings
        g.negative_prompt_embeds = negative_text_embeddings
        g.pooled_prompt_embeds = pooled_text_embeddings
        g.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        
