from evolution.image_grid import ImageGridViewer
import tkinter as tk
import random
from abc import ABC, abstractmethod
import util.common_settings as common_settings

class Evolver(ABC):
    def __init__(self, args, population_size = 9):
        self.args = args
        self.population_size = population_size
        self.steps = common_settings.NUM_INFERENCE_STEPS
        self.guidance_scale = common_settings.GUIDANCE_SCALE

        self.evolution_history = []
        self.generation = 0

    def get_generation(self):
        return self.generation

    def start_evolution(self, allow_prompt=False, allow_negative_prompt=False):
        self.genomes = []
        self.generation = 0

        self.root = tk.Tk()
        self.viewer = ImageGridViewer(
            self.root, 
            callback_fn=self.next_generation,
            back_fn=self.previous_generation,
            generation_fn=self.get_generation,
            allow_prompt=allow_prompt,
            allow_negative_prompt=allow_negative_prompt,
            args = self.args
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

    def next_generation(self,selected_images,prompt=None,negative_prompt=None):
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
                g.prompt = prompt
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

