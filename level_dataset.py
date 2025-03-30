import json
import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
from tokenizer import Tokenizer
import os
import matplotlib
import matplotlib.pyplot as plt

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

    def _augment_scene_and_caption(self, scene, caption): # augments by flipping
        scene_tensor = torch.flip(scene, dims=[-1]) # Had to make -1 to work, which seems odd, but results look right
        caption_tensor = torch.tensor(self._swap_caption_tokens(caption), dtype=torch.long)

        return scene_tensor, caption_tensor

    def _shuffle_data(self):
        """Shuffles the dataset."""
        combined = list(zip(self.data, self.tokenized_captions))
        random.shuffle(combined)
        self.data, self.tokenized_captions = zip(*combined)

    def _pad_sequence(self, tokens):
        """Pads tokenized sequences to max_length with a padding token (assumed to be '[PAD]')."""
        pad_token = self.tokenizer.token_to_id["[PAD]"]
        return tokens + [pad_token] * (self.max_length - len(tokens))

    def _swap_caption_tokens(self, caption_tensor):
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

    def get_batch(self, idx):
        """
        Retrieves a batch of data.

        Args:
            idx (int): Batch index.

        Returns:
            - In "mlm" mode: token_tensor (Tensor) of shape (batch_size, max_length)
            - In "diffusion" mode: (scenes_tensor, captions_tensor)
              scenes_tensor is one-hot encoded with shape (batch_size, num_tiles, height, width)
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
        scene_tensor = torch.tensor(batch_scenes, dtype=torch.long)
        
        # Apply augmentation if enabled
        if self.augment:
            for i in range(len(batch_scenes)):
                if random.choice([True, False]):  # Randomly decide whether to flip
                    # Convert to tensor for consistent handling
                    scene_tensor[i], caption_tensor[i] = self._augment_scene_and_caption(scene_tensor[i], batch_tokens[i])

        # Convert to one-hot encoding for diffusion model
        one_hot_scenes = F.one_hot(scene_tensor, num_classes=self.num_tiles).float()
        # Permute dimensions to [batch_size, num_tiles, height, width]
        one_hot_scenes = one_hot_scenes.permute(0, 3, 1, 2)

        return one_hot_scenes, caption_tensor

    def __len__(self):
        """Returns number of batches."""
        return (len(self.tokenized_captions) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        """
        Fetches one sample.

        Returns:
            - In "mlm" mode: tokenized caption
            - In "diffusion" mode: (scene_tensor, caption_tensor)
              scene_tensor is one-hot encoded with shape (num_tiles, height, width)
        """
        sample = self.data[idx]
        augmented_caption = self._augment_caption(sample["caption"])
        caption_tokens = self.tokenizer.encode(augmented_caption)
        caption_tokens = self._pad_sequence(caption_tokens)
        caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)

        if self.mode == "mlm":
            return caption_tensor  # MLM only uses captions

        scene_tensor = torch.tensor(sample["scene"], dtype=torch.long)  # Convert scene to tensor
        
        # Apply augmentation if enabled
        if self.augment and random.choice([True, False]):
            scene_tensor, caption_tensor = self._augment_scene_and_caption(scene_tensor, caption_tokens)

        # Convert to one-hot encoding for diffusion model
        one_hot_scene = F.one_hot(scene_tensor, num_classes=self.num_tiles).float()
        # Permute dimensions to [num_tiles, height, width]
        one_hot_scene = one_hot_scene.permute(2, 0, 1)

        return one_hot_scene, caption_tensor

    def decode_caption(self, token_ids):
        """Converts a sequence of token IDs back into a readable caption."""
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self):
        """Returns the size of the tokenizer vocabulary."""
        return len(self.tokenizer.get_vocab())

    def get_sample_caption(self, idx):
        """Returns the raw caption from the dataset for debugging."""
        return self.data[idx]["caption"]

    def decode_scene(self, one_hot_scene):
        """
        Converts a one-hot encoded level scene tensor back to the original list of lists of integers.
    
        Args:
            one_hot_scene (Tensor): One-hot encoded scene tensor with shape [num_tiles, height, width]
                                   or [batch_size, num_tiles, height, width] if batched
    
        Returns:
            List of lists of integers representing the original scene layout
        """
        # Check if we have a batched input
        is_batched = len(one_hot_scene.shape) == 4
    
        if is_batched:
            # For batched input, we'll just process the first example
            # You could extend this to process all examples if needed
            one_hot_scene = one_hot_scene[0]
    
        # Permute back to [height, width, num_tiles] format
        one_hot_permuted = one_hot_scene.permute(1, 2, 0)
    
        # Get the indices (tile IDs) where the one-hot encoding has a 1
        scene_indices = torch.argmax(one_hot_permuted, dim=2)
    
        # Convert to a list of lists
        scene_list = scene_indices.tolist()
    
        return scene_list

    def visualize_samples(self, samples, output_dir, subdir):
        """
        Visualize generated samples and save as images.
    
        Args:
            samples: One-hot encoded samples from the diffusion model
            output_dir: Directory to save visualizations
    
        Returns:
            List of tile index maps for the samples
        """
        # Create directory for the samples
        samples_dir = os.path.join(output_dir, subdir)
        os.makedirs(samples_dir, exist_ok=True)
    
        # Convert from one-hot to tile indices
        sample_indices = []
        plt.figure(figsize=(16, 4))
    
        for i, sample in enumerate(samples):
            # Convert one-hot back to indices (get most likely tile for each position)
            # [num_tiles, height, width] -> [height, width]
            sample_index = torch.argmax(sample, dim=0).cpu().numpy()
            sample_indices.append(sample_index)
        
            # Plot and save
            plt.subplot(1, 4, i + 1)
            plt.imshow(sample_index, cmap='viridis')
            plt.colorbar(label='Tile Type')
            plt.title(f"Sample {i+1}")
    
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, "samples_grid.png"))
        plt.close()
    
        # Save individual samples
        for i, sample_index in enumerate(sample_indices):
            plt.figure(figsize=(8, 8))
            plt.imshow(sample_index, cmap='viridis')
            plt.colorbar(label='Tile Type')
            plt.title(f"Sample {i+1}")
            plt.savefig(os.path.join(samples_dir, f"sample_{i}.png"))
            plt.close()
    
        return sample_indices

if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenizer.load('SMB1_Tokenizer.pkl')

    # Create MLM dataset
    mlm_dataset = LevelDataset('SMB1_LevelsAndCaptions.json', tokenizer, batch_size=16, mode="mlm", random_seed=9999)
    batch = mlm_dataset.get_batch(0)
    print("MLM Batch Shape:", batch.shape)  # Should be (16, max_length)

    print(batch[0])
    print(mlm_dataset.tokenizer.decode(batch[0].tolist()))

    # Create Diffusion dataset
    diffusion_dataset = LevelDataset('SMB1_LevelsAndCaptions.json', tokenizer, batch_size=16, mode="diffusion", random_seed=9999)
    scenes, captions = diffusion_dataset.get_batch(0)
    print("Diffusion Batch Shapes:", scenes.shape, captions.shape) 

    print(scenes[0])
    print(torch.tensor(diffusion_dataset.decode_scene(scenes[0])))
    print(diffusion_dataset.tokenizer.decode(captions[0].tolist()))

    print("-----------")

    scene, caption = diffusion_dataset[100]
    print(torch.tensor(diffusion_dataset.decode_scene(scene)))
    scene, caption = diffusion_dataset._augment_scene_and_caption(scene, caption)
    print(torch.tensor(diffusion_dataset.decode_scene(scene)))

    print("-----------")
    scene, caption = diffusion_dataset[100]
    print(scene)

    diffusion_dataset.visualize_samples(diffusion_dataset.get_batch(0)[0][0:4], "TEST", "TEMP")