from image_grid import ImageGridViewer
import tkinter as tk
import random
from genome import DiffusionGenome
import torch
from abc import ABC, abstractmethod
from level_dataset import visualize_samples
#from text_diffusion_pipeline import TextConditionalDDPMPipeline
#from diffusers import DDPMPipeline
from latent_diffusion_pipeline import UnconditionalDDPMPipeline
from level_dataset import visualize_samples, convert_to_level_format
from caption_match import compare_captions
from create_ascii_captions import assign_caption, extract_tileset
import numpy as np
import argparse

class Evolver(ABC):
    def __init__(self, population_size = 9):
        self.population_size = population_size
        self.steps = 50
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

    def next_generation(self,selected_images):
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


class DiffusionEvolver(Evolver):
    def __init__(self, model_path, width, tileset_path='..\TheVGLC\Super Mario Bros\smb.json'):
        Evolver.__init__(self)

        self.width = width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = UnconditionalDDPMPipeline.from_pretrained(model_path).to(self.device)

        _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)

    def random_latent(self, seed=1):
        # Create the initial noise latents (this is what the pipeline does internally)
        height = 16
        width = self.width
        num_channels_latents = len(self.id_to_char)
        latents_shape = (1, num_channels_latents, height, width)
        latents = torch.randn(
            latents_shape, 
            generator=torch.manual_seed(seed)        
        ).to("cpu")
        return latents

    def initialize_population(self):
        self.genomes = [DiffusionGenome(self.width, seed, self.steps, self.guidance_scale, latents=self.random_latent(seed)) for seed in range(self.population_size)]

    def generate_image(self, g):
        # generate fresh new image
        print(f"Generate new image for {g}")
        generator = torch.Generator("cuda").manual_seed(g.seed)

        settings = {
            "batch_size" : 1,
            # "guidance_scale" : g.guidance_scale, # Remove this from genome?
            "num_inference_steps" : g.num_inference_steps,
            # "strength" : g.strength, # Definitely don't need this
            "output_type" : "tensor",
            "latents" : g.latents.to("cuda")
        }
        
        images = self.pipe(
            generator=generator,
            **settings
        ).images

        g.latents.to("cpu")

        images = torch.tensor(images).permute(0, 3, 1, 2)  # Convert (B, H, W, C) -> (B, C, H, W)

        # Convert to indices
        #sample_tensor = torch.tensor(images[0])
        #sample_indices = convert_to_level_format(sample_tensor)
        
        # Add level data to the list
        #scene = sample_indices[0].tolist() # Always just one scene: (1,16,16)
 
        # actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False) # self.args.describe_locations, self.args.describe_absence)

        #print(f"Describe resulting image: {actual_caption}")
        #compare_score = compare_captions(self.prompt, actual_caption)
        #print(f"Comparison score: {compare_score}")

        return visualize_samples(images)

def parse_args():
    parser = argparse.ArgumentParser(description="Evolve levels with unconditional diffusion model")    
    # Model and generation parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained diffusion model")
    parser.add_argument("--tileset_path", default='..\TheVGLC\Super Mario Bros\smb.json', help="Descriptions of individual tile types")
    parser.add_argument("--describe_locations", action="store_true", default=False, help="Include location descriptions in the captions")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--width", type=int, default=16, help="Tile width of generated level")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evolver = DiffusionEvolver(args.model_path, args.width, args.tileset_path)
    evolver.start_evolution()