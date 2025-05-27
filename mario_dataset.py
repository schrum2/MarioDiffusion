import json
import torch
import random
from torch.utils.data import Dataset
from collections import Counter
import matplotlib.pyplot as plt
import math

class MarioPatchDataset(Dataset):
    def __init__(self, json_path, patch_size, ignore_tile_id=-1, subsample_threshold=0.001):
        with open(json_path, 'r') as f:
            self.patches = json.load(f)

        if isinstance(patch_size, list):
            patch_size = len(patch_size)
        self.patch_size = int(patch_size)
        self.ignore_tile_id = ignore_tile_id
        self.subsample_threshold = subsample_threshold

        total_tiles = self.patch_size * self.patch_size
        self.center_idx = total_tiles // 2

        # Validate patch dimensions
        if not all(len(patch) == patch_size and all(len(row) == patch_size for row in patch)
                   for patch in self.patches):
            raise ValueError(f"All patches must be {patch_size}x{patch_size}")

        self.center_counts = self._count_center_frequencies()
        self.sampling_probs = self._compute_subsampling_probs()
        self.samples = self._filter_patches()

        # Useful for diagnostics, but not strictly required each time
        #self.plot_center_distribution("center_tile_distribution.png")

        print(f"Loaded {len(self.samples)} valid {patch_size}x{patch_size} patches")

    def plot_center_distribution(self, save_path=None):
        counts = Counter(center for center, _ in self.samples)
        tiles, freqs = zip(*sorted(counts.items()))
        
        plt.figure(figsize=(10, 4))
        plt.bar(tiles, freqs)
        plt.xlabel("Tile ID (center)")
        plt.ylabel("Frequency (after subsampling)")
        plt.title("Center Tile Distribution After Subsampling")
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def _count_center_frequencies(self):
        counts = Counter()
        for patch in self.patches:
            flat = [tile for row in patch for tile in row]
            center = flat[self.center_idx]
            if center != self.ignore_tile_id:
                counts[center] += 1
        return counts

    def _compute_subsampling_probs(self):
        total = sum(self.center_counts.values())
        probs = {}
        for token, freq in self.center_counts.items():
            f = freq / total
            prob = (math.sqrt(f / self.subsample_threshold) + 1) * (self.subsample_threshold / f)
            # Clamp to [0, 1]
            probs[token] = min(prob, 1.0)
        return probs

    def _filter_patches(self):
        valid = []
        for patch in self.patches:
            flat = [tile for row in patch for tile in row]
            center = flat[self.center_idx]
            context = flat[:self.center_idx] + flat[self.center_idx + 1:]

            if center == self.ignore_tile_id:
                continue
            filtered_context = [t for t in context if t != self.ignore_tile_id]
            if not filtered_context:
                continue

            # Subsampling based on frequency
            keep_prob = self.sampling_probs.get(center, 1.0)
            if random.random() > keep_prob:
                continue

            valid.append((center, filtered_context))
        return valid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        center, context = self.samples[idx]
        center = torch.tensor(center, dtype=torch.long)
        context = torch.tensor(context, dtype=torch.long)
        return center, context
