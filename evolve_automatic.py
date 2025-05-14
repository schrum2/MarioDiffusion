"""
    This will start as just a script for evolving diffusion models,
    but hopefully it can generalize to GANs too.
"""

import numpy as np
from cmaes import CMA
import argparse

def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=50, help="Number of generations to run")
    parser.add_argument("--width", type=int, default=16, help="Width of a generated scene (not used in quadratic demo)")
    args = parser.parse_args()

    optimizer = CMA(mean=np.zeros(2), sigma=1.3)

    for generation in range(args.generations):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)