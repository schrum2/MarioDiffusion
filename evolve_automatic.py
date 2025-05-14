"""
    This will start as just a script for evolving diffusion models,
    but hopefully it can generalize to GANs too.

    Cite: https://pypi.org/project/cmaes/
"""

import numpy as np
from cmaes import CMA
import argparse

def caption_fitness(x):
    """
    Generate the scene, then generate its caption, and compare to a desired caption
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=50, help="Number of generations to run")
    parser.add_argument("--width", type=int, default=16, help="Width of a generated scene")
    parser.add_argument("--height", type=int, default=16, help="Height of a generated scene")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of possible tiles/channels")

    parser.add_argument("--model_path", type=str, required=True, help="Path to model whose latent space will be explored")
    parser.add_argument("--target_caption", type=str, required=True, help="Caption that scenes will be evolved to match")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--num_diffusion_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for diffusion model")
    parser.add_argument("--population_size", type=int, default=10, help="Number of genomes in the population")

    args = parser.parse_args()

    W = args.width
    H = args.height
    C = args.num_tiles

    optimizer = CMA(mean=np.zeros(W*H*C), sigma=1.3, population_size=args.population_size, seed=args.seed)

    for generation in range(args.generations):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = caption_fitness(x)
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)