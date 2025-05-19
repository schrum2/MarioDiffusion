import json
import torch
from torch.utils.data import Dataset

class MarioPatchDataset(Dataset):
    def __init__(self, json_path, ignore_tile_id=-1):
        with open(json_path, 'r') as f:
            self.patches = json.load(f)

        self.ignore_tile_id = ignore_tile_id
        self.samples = self._filter_patches()

    def _filter_patches(self):
        valid = []
        for patch in self.patches:
            flat = [tile for row in patch for tile in row]
            center = flat[4]
            context = flat[:4] + flat[5:]
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
