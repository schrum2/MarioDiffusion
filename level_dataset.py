import json
import torch
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
from tokenizer import Tokenizer


class LevelDataset:
    def __init__(self, json_path, tokenizer, batch_size=32, shuffle=True, max_length=None, mode="diffusion", random_seed=1, augment=True, limit=-1, num_tiles=15):
        """
            Args:
            json_path (str): Path to JSON file with captions.
            tokenizer (Tokenizer): Tokenizer instance.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle data at the start of an epoch.
            max_length (int, optional): Maximum length for tokenized captions.
            mode (str): "diffusion" for level scenes + captions, "mlm" for masked language model training.
            augment (bool): Whether to apply data augmentation
            limit (int): restrict dataset to this size if not -1
            num_tiles (int): Number of different tile types for one-hot encoding
        """
        assert mode in ["mlm", "diffusion"], "Mode must be 'mlm' or 'diffusion'."

        self.random_seed = random_seed
        random.seed(self.random_seed)  # Ensure reproducibility

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.augment = augment
        self.num_tiles = num_tiles

        # Load data
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        if limit > -1:
            # Random selection of limited portion of data (if limit is less than actual size)
            self.data = random.sample(self.data, limit)

        print(f"Training samples: {len(self.data)}")

        # Tokenize all captions in advance
        self.tokenized_captions = [self.tokenizer.encode(entry["caption"]) for entry in self.data]

        # Ensure all tokenized captions are lists of integers
        for i, tokens in enumerate(self.tokenized_captions):
            if not all(isinstance(token, int) for token in tokens):
                raise ValueError(f"Tokenization error at index {i}: {tokens}")

        # Determine padding length (if not provided)
        if self.max_length is None:
            self.max_length = max(len(tokens) for tokens in self.tokenized_captions)

        # Shuffle dataset
        if self.shuffle:
            self._shuffle_data()

    def _augment_caption(self, caption):
        """Shuffles period-separated phrases in the caption."""
        if self.augment:
            phrases = caption[:-1].split(". ") # [:-1] removes the last period
            random.shuffle(phrases)  # Shuffle phrases
            return ". ".join(phrases) + "."
        else:
            return caption # Same as original

    def _shuffle_data(self):
        """Shuffles the dataset."""
        combined = list(zip(self.data, self.tokenized_captions))
        random.shuffle(combined)
        self.data, self.tokenized_captions = zip(*combined)

    def _pad_sequence(self, tokens):
        """Pads tokenized sequences to max_length with a padding token (assumed to be '[PAD]')."""
        pad_token = self.tokenizer.token_to_id["[PAD]"]
        return tokens + [pad_token] * (self.max_length - len(tokens))

    def flip_scene_horizontally(self, scene):
        """Flip scene horizontally, working with either tensor or list representation"""
        if isinstance(scene, torch.Tensor):
            # If scene is already a tensor (potentially one-hot encoded)
            if scene.dim() == 2:  # Regular 2D scene (H×W)
                return torch.flip(scene, dims=[1])
            elif scene.dim() == 3:  # One-hot encoded scene (C×H×W)
                return torch.flip(scene, dims=[2])
        else:
            print(scene)
            raise ValueError("Why is data still a list of lists?")
            # Should not ever happen
            # Original list of lists implementation
            # return [row[::-1] for row in scene]

    def swap_caption_tokens(self, caption_tensor):
        left_id = self.tokenizer.token_to_id["left"]
        right_id = self.tokenizer.token_to_id["right"]
        ascending_id = self.tokenizer.token_to_id["ascending"]
        descending_id = self.tokenizer.token_to_id["descending"]
        
        swapped_caption = []
        for token in caption_tensor:
            if token == left_id:
                swapped_caption.append(right_id)
            elif token == right_id:
                swapped_caption.append(left_id)
            elif token == ascending_id:
                swapped_caption.append(descending_id)
            elif token == descending_id:
                swapped_caption.append(ascending_id)
            else:
                swapped_caption.append(token)
        
        return swapped_caption

    def _prepare_scene_for_diffusion(self, scene):
        """Convert scene to one-hot encoding for diffusion model"""
        if self.mode == "diffusion":
            # Convert to tensor if not already
            if not isinstance(scene, torch.Tensor):
                scene = torch.tensor(scene, dtype=torch.long)
            
            # Convert to one-hot encoding
            one_hot = F.one_hot(scene, num_classes=self.num_tiles).float()
            
            # Permute dimensions to [C, H, W] format for diffusion model
            # From [H, W, C] to [C, H, W]
            return one_hot.permute(2, 0, 1)
        else:
            # Should not be called in MLM mode
            raise ValueError("Should not be called in MLM mode")

    def get_batch(self, idx):
        """
        Retrieves a batch of data.

        Args:
            idx (int): Batch index.

        Returns:
            - In "mlm" mode: token_tensor (Tensor) of shape (batch_size, max_length)
            - In "diffusion" mode: (pixel_values, captions_tensor) where pixel_values is 
              one-hot encoded [batch_size, num_tiles, height, width]
        """
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.data))

        batch_tokens = [self.tokenizer.encode(self._augment_caption(self.data[i]["caption"])) for i in range(start, end)]
        batch_tokens = [self._pad_sequence(tokens) for tokens in batch_tokens]
        caption_tensor = torch.tensor(batch_tokens, dtype=torch.long)

        if self.mode == "mlm":
            return caption_tensor  # MLM only uses captions

        # Get level scenes for diffusion training
        batch_scenes = [self.data[i]["scene"] for i in range(start, end)]
        
        # Convert scenes to tensors
        batch_scene_tensors = [torch.tensor(scene, dtype=torch.long) for scene in batch_scenes]
        
        # Apply data augmentation
        if self.augment:
            for i in range(len(batch_scene_tensors)):
                if random.choice([True, False]):  # Randomly decide whether to flip
                    batch_scene_tensors[i] = self.flip_scene_horizontally(batch_scene_tensors[i])
                    batch_tokens[i] = self.swap_caption_tokens(batch_tokens[i])
        
        # Convert to one-hot encoding for diffusion
        one_hot_scenes = []
        for scene_tensor in batch_scene_tensors:
            one_hot = F.one_hot(scene_tensor, num_classes=self.num_tiles).float()
            # Convert from [H, W, C] to [C, H, W]
            one_hot_scenes.append(one_hot.permute(2, 0, 1))
        
        # Stack into a batch
        scene_tensor = torch.stack(one_hot_scenes)
        caption_tensor = torch.tensor(batch_tokens, dtype=torch.long)

        return scene_tensor, caption_tensor

    def __len__(self):
        """Returns number of batches."""
        return (len(self.tokenized_captions) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        """
        Fetches one sample.

        Returns:
            - In "mlm" mode: tokenized caption
            - In "diffusion" mode: (scene_tensor, caption_tensor)
              where scene_tensor is one-hot encoded [C, H, W]
        """
        sample = self.data[idx]
        augmented_caption = self._augment_caption(sample["caption"])
        caption_tokens = self.tokenizer.encode(augmented_caption)
        caption_tokens = self._pad_sequence(caption_tokens)
        caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)

        if self.mode == "mlm":
            return caption_tensor  # MLM only uses captions

        scene_tensor = torch.tensor(sample["scene"], dtype=torch.long)
        
        # Apply augmentation
        if self.augment and random.choice([True, False]):
            scene_tensor = self.flip_scene_horizontally(scene_tensor)
            caption_tensor = torch.tensor(self.swap_caption_tokens(caption_tokens.tolist()), dtype=torch.long)
        
        # Prepare for diffusion
        scene_tensor = self._prepare_scene_for_diffusion(scene_tensor)

        return scene_tensor, caption_tensor

    def decode_caption(self, token_ids):
        """Converts a sequence of token IDs back into a readable caption."""
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self):
        """Returns the size of the tokenizer vocabulary."""
        return len(self.tokenizer.get_vocab())

    def get_sample_caption(self, idx):
        """Returns the raw caption from the dataset for debugging."""
        return self.data[idx]["caption"]