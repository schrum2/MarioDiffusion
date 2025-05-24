import json
import torch
from torch.utils.data import Dataset

class MarioPatchDataset(Dataset):
    def __init__(self, json_path, patch_size, ignore_tile_id=-1):
        with open(json_path, 'r') as f:
            self.patches = json.load(f)

        if isinstance(patch_size, list):
            patch_size = len(patch_size)
        self.patch_size = int(patch_size)
        self.ignore_tile_id = ignore_tile_id

        # Calculate center position for any size patch
        total_tiles = self.patch_size * self.patch_size
        self.center_idx = (total_tiles // 2)

        
        # Validate patch dimensions
        if not all(len(patch) == patch_size and all(len(row) == patch_size for row in patch) 
                  for patch in self.patches):
            raise ValueError(f"All patches must be {patch_size}x{patch_size}")
            
        self.samples = self._filter_patches()
        print(f"Loaded {len(self.samples)} valid {patch_size}x{patch_size} patches")

    def _filter_patches(self):
        valid = []
        for patch in self.patches:
            flat = [tile for row in patch for tile in row]

            # Get center tile 
            center = flat[self.center_idx]

            # Get context (all tiles except center)
            context = flat[:self.center_idx] + flat[self.center_idx + 1:]

            # Filter invalid tiles
            if center == self.ignore_tile_id:
                continue

            filtered_context = [t for t in context if t != self.ignore_tile_id]
            if not filtered_context:
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