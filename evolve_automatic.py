"""
    This will start as just a script for evolving diffusion models,
    but hopefully it can generalize to GANs too.

    Cite: https://pypi.org/project/cmaes/
"""

import numpy as np
from cmaes import CMA
import argparse
import torch

from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from level_dataset import visualize_samples, convert_to_level_format
from captions.caption_match import process_scene_segments
from create_ascii_captions import extract_tileset
import os
import util.common_settings as common_settings

def caption_fitness(x):
    """
    Generate the scene, then generate its caption, and compare to a desired caption
    """
    global W, H, C, args
    # Convert x to a scene representation
    latent_input = torch.tensor(x.reshape((1, C, H, W)), dtype=torch.float32)
    # Currently seed matches simulation seed, but a fixed seed could be carried with each genome
    generator = torch.Generator("cuda").manual_seed(args.seed)

    settings = {
        "guidance_scale": args.guidance_scale, 
        "num_inference_steps": args.num_inference_steps,
        "output_type": "tensor",
        "raw_latent_sample": latent_input.to("cuda")
    }
        
    # Include caption if desired
    if True: # Make this a check of whether the model supports text embedding
        settings["caption"] = args.target_caption
        
    images = pipe(
        generator=generator,
        **settings
    ).images

    latent_input.to("cpu")
    # Convert to indices
    sample_indices = convert_to_level_format(images)
        
    # Add level data to the list
    scene = sample_indices[0].tolist() # Always just one scene: (1,16,16)
    global id_to_char, char_to_id, tile_descriptors
    
    # This would be for whole scene
    #actual_caption = assign_caption(scene, id_to_char, char_to_id, tile_descriptors, False, args.describe_absence)

    average_score, segment_captions, segment_scores = process_scene_segments(
        scene=scene,
        segment_width=W,
        prompt=args.target_caption,
        id_to_char=id_to_char,
        char_to_id=char_to_id,
        tile_descriptors=tile_descriptors,
        describe_locations=False,
        describe_absence=args.describe_absence,
        verbose=False
    )

    #print(latent_input)
    #input("next")
    fitness = -average_score

    global best_fitness
    # Check if the caption is new and if it has a better score
    #print(seen_captions)
    for c in segment_captions:
        if best_fitness > fitness:
            visualize_samples(images).show()
            print(fitness, segment_captions)
            input("Press enter for next")
            seen_captions.add(c)            
            best_fitness = fitness

    # Negative score is better, since CMA-ES wants to minimize
    return fitness, segment_captions

class SimpleEvolutionaryOptimizer:
    """
    A simple evolutionary optimizer placeholder class.
    Implement ask() and tell() methods as needed.
    """
    def __init__(self, population_size, seed=None, mutation_sigma=5.0, tournament_size=2):
        self.mutation_sigma = mutation_sigma
        self.tournament_size = tournament_size
        self.population_size = population_size
        self.seed = seed
        # Add any additional initialization here
        global W, H, C
        self.population = np.random.rand(population_size, W * H * C)

    def ask(self):
        """
        Generate a new candidate solution.
        Placeholder: implement logic for generating candidates.
        """
        return self.population

    def tell(self, solutions):
        """
        Update the optimizer with evaluated solutions.
        Placeholder: implement logic for updating population.
        """
        # solutions: list of (individual, fitness)
        individuals = np.array([ind for ind, fit in solutions])
        fitnesses = np.array([fit for ind, fit in solutions])

        # 10% elitism: keep the top 10% individuals unchanged
        elite_count = 0 # max(1, int(0.1 * self.population_size))
        #elite_indices = np.argsort(fitnesses)[:elite_count]
        #elites = individuals[elite_indices]

        new_population = [] # [elites[i] for i in range(elite_count)]

        # Fill the rest of the population with mutated children
        # Population_size is elite and children
        for _ in range(self.population_size - elite_count):
            # Tournament selection
            idxs = np.random.choice(len(individuals), self.tournament_size, replace=False)
            best_idx = idxs[np.argmin(fitnesses[idxs])]  # assuming lower fitness is better
            parent = individuals[best_idx]
            # Gaussian perturbation (mutation)
            child = parent + np.random.normal(0, self.mutation_sigma, size=parent.shape)
            new_population.append(child)
        self.population = np.array(new_population)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=50, help="Number of generations to run")
    parser.add_argument("--width", type=int, default=common_settings.MARIO_WIDTH, help="Width of a generated scene")
    parser.add_argument("--height", type=int, default=common_settings.MARIO_HEIGHT, help="Height of a generated scene")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of possible tiles/channels")

    parser.add_argument("--model_path", type=str, required=True, help="Path to model whose latent space will be explored")
    parser.add_argument("--target_caption", type=str, required=True, help="Caption that scenes will be evolved to match")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--num_inference_steps", type=int, default=common_settings.NUM_INFERENCE_STEPS, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=common_settings.GUIDANCE_SCALE, help="Guidance scale for diffusion model")
    parser.add_argument("--population_size", type=int, default=10, help="Number of genomes in the population")

    parser.add_argument("--tileset", default=os.path.join('..', 'TheVGLC', 'Super Mario Bros', 'smb.json'), help="Descriptions of individual tile types")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")

    global args
    args = parser.parse_args()

    global W, H, C
    W = args.width
    H = args.height
    C = args.num_tiles

    global pipe
    pipe = TextConditionalDDPMPipeline.from_pretrained(args.model_path).to("cuda")

    global id_to_char, char_to_id, tile_descriptors
    _, id_to_char, char_to_id, tile_descriptors = extract_tileset(args.tileset)

    global seen_captions
    seen_captions = set()

    global best_fitness
    best_fitness = float("inf") # Trying to minimize

    # optimizer = CMA(mean=np.zeros(W*H*C), sigma=5.0, population_size=args.population_size, seed=args.seed)
    # Replace CMA with a simple evolutionary optimizer
    optimizer = SimpleEvolutionaryOptimizer(
        population_size=args.population_size,
        seed=args.seed,
        mutation_sigma=0.1,
        tournament_size=2
    )

    for generation in range(args.generations):
        solutions = []
        for x in optimizer.ask():
            #x = optimizer.ask()
            value, caption = caption_fitness(x)
            solutions.append((x, value))
            print(f"#{generation} {value}:{caption}")
        optimizer.tell(solutions)