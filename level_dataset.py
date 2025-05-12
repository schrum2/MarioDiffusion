import json
import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
from tokenizer import Tokenizer
import os
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
import io
from PIL import Image

# Global variable to store the loaded sprite sheet
_sprite_sheet = None

def samples_to_scenes(all_samples):
    # Convert to list
    samples_list = [all_samples[i] for i in range(len(all_samples))]
    scenes = []
    # Process and collect individual samples
    for _, sample in enumerate(samples_list):
        # Convert to indices
        sample_tensor = sample.unsqueeze(0) # if sample.shape[0] == args.num_tiles else sample
        sample_indices = convert_to_level_format(sample_tensor)
        
        # Add level data to the list
        scene = sample_indices[0].tolist() # Always just one scene: (1,16,16)
        scenes.append(scene)

    return scenes

def convert_to_level_format(sample):
    """Convert model output to level indices"""
    sample_indices = torch.argmax(sample, dim=1).cpu().numpy()
    #print(sample_indices.shape)
    return sample_indices

def get_pil_image_from_plt(fig):
    """
    Converts a matplotlib figure to a PIL Image.

    Args:
        fig: The matplotlib Figure object.

    Returns:
        A PIL Image object representing the figure.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img

def colors():
    # Create custom colormap for integers 0-15
    colorslist = [
        (0.8, 0.9, 1.0),    # 0 = very light blue: sky
        (0.0, 0.4, 0.0),    # 1 = dark green: left upper lip of pipe
        (0.0, 0.2, 0.0),    # 2 = darker green: right upper lip of pipe
        (1.0, 0.7, 0.9),    # 3 = pink: question block with power up
        (0.0, 0.0, 0.0),    # 4 = black: Cannon head
        (1.0, 0.0, 0.0),    # 5 = bright red: enemy
        (0.6, 0.4, 0.0),    # 6 = dark gold: question block with coin
        (0.8, 0.4, 0.0),    # 7 = dark orange: breakable brick block
        (0.5, 0.2, 0.1),    # 8 = brownish red: solid block/floor
        (0.6, 0.9, 0.6),    # 9 = light green: left edge of pipe body
        (0.7, 1.0, 0.7),    # 10 = lighter green: right edge of pipe body
        (0.5, 0.5, 0.5),    # 11 = grey: Cannon support
        (1.0, 1.0, 0.0),    # 12 = yellow: coin
        (1.0, 1.0, 1.0),    # 13 = white
        (0.6, 0.0, 0.9),    # 14 = violet
        (0.3, 0.3, 0.3)     # 15 (extra color just in case)
    ]

    return colorslist

def tiles():
    """
    Maps integers 0-15 to 16x16 pixel sprites from mapsheet.png.

    Returns:
        A list of 16x16 pixel tile images.
    """
    global _sprite_sheet

    # Load the sprite sheet only once
    if _sprite_sheet is None:
        _sprite_sheet = Image.open("mapsheet.png")

    # Hardcoded coordinates for the first 16 tiles (row, col)
    tile_coordinates = [
        (2,5),    # 0 = Sky
        (2,2),    # 1 = left upper lip of pipe
        (3,2),    # 2 = right upper lip of pipe
        (0,1),    # 3 = question block with power up
        (3,0),    # 4 = Cannon head
        (7,4),    # 5 = enemy
        (2,1),    # 6 = question block with coin
        (2,6),    # 7 = breakable brick block
        (1,0),    # 8 = solid block/floor
        (4,2),    # 9 = left edge of pipe body
        (5,2),    # 10 = right edge of pipe body
        (4,0),    # 11 = Cannon support (should be 5,0 sometimes?)
        (7,1),    # 12 = coin
        (0,1),    # 13 = Nothing
        (0,6),    # 14 = Nothing
        (1,6)     # 15 = Nothing (extra just in case)
    ]

    # Extract each tile as a 16x16 image
    tile_images = []
    for col, row in tile_coordinates:
        left = col * 16
        upper = row * 16
        right = left + 16
        lower = upper + 16
        tile = _sprite_sheet.crop((left, upper, right, lower))
        tile_images.append(tile)

    return tile_images

def visualize_samples(samples, output_dir=None, use_tiles=True):
    """
    Visualize generated samples and save as images.

    Args:
        samples: One-hot encoded samples from the diffusion model: [samples, channels, height, width]
        output_dir: Directory to save visualizations
        use_tiles: If True, use tile images instead of colors for visualization

    Returns:
        List of tile index maps for the samples
    """
    if len(samples.shape) != 4:
        print(samples.shape)
        raise ValueError("Shape of input should be [samples, channels, height, width]")
    if samples.shape[1] != 15:
        print(samples.shape)
        raise ValueError("Hard coded for 15 channels (change code to generalize beyond Mario)")

    # Create directory for the samples
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert from one-hot to tile indices
    sample_indices = []
    num_samples = len(samples)
    grid_cols = min(4, num_samples)  # Limit to 4 columns
    grid_rows = (num_samples + grid_cols - 1) // grid_cols  # Calculate rows needed

    if use_tiles:
        tile_images = tiles()
        tile_size = 16
        for i, sample in enumerate(samples):
            sample_index = torch.argmax(sample, dim=0).cpu().numpy()
            sample_indices.append(sample_index)

            # Create a blank image to hold the tile-based visualization
            height, width = sample_index.shape
            composite_image = Image.new('RGB', (width * tile_size, height * tile_size))

            for row in range(height):
                for col in range(width):
                    tile_id = sample_index[row, col]
                    tile_image = tile_images[tile_id]
                    composite_image.paste(tile_image, (col * tile_size, row * tile_size))

            if output_dir:
                composite_image.save(os.path.join(output_dir, f"sample_{i}.png"))
            else:
                return composite_image

    else:
        # Create custom colormap for integers 0-15
        colorslist = colors()
        custom_cmap = matplotlib.colors.ListedColormap(colorslist[:15])

        plt.figure(figsize=(4 * grid_cols, 4 * grid_rows))  # Adjust figure size dynamically

        for i, sample in enumerate(samples):
            sample_index = torch.argmax(sample, dim=0).cpu().numpy()
            sample_indices.append(sample_index)

            # Plot and save
            plt.subplot(grid_rows, grid_cols, i + 1)
            plt.imshow(sample_index, cmap=custom_cmap, vmin=0, vmax=14)  # Set vmin and vmax to ensure color mapping
            plt.title(f"Sample {i+1}")

        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, "samples_grid.png"))
            result = None
        else:
            result = get_pil_image_from_plt(plt.gcf())

        plt.close()

        # Returning an image instead of saving many images
        if result:
            return result

    return sample_indices

class LevelDataset(Dataset):
    def __init__(self, json_path, tokenizer, shuffle=True, max_length=None, mode="diffusion", augment=True, limit=-1, num_tiles=15):
        """
            Args:
            json_path (str): Path to JSON file with captions.
            tokenizer (Tokenizer): Tokenizer instance.
            shuffle (bool): Whether to shuffle data at the start of an epoch.
            max_length (int, optional): Maximum length for tokenized captions.
            mode (str): "diffusion" for level scenes + captions, "mlm" for masked language model training.
            augment (bool): Whether to apply data augmentation
            limit (int): restrict dataset to this size if not -1
            num_tiles (int): Number of different tile types for one-hot encoding
        """
        assert mode in ["mlm", "diffusion"], "Mode must be 'mlm' or 'diffusion'."

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
        """
            swapping directional tokens for consistency with flipped scenes
            scene: list of lists of integers level scene representation
        """

        if len(scene.shape) != 2:
            print(scene)
            raise ValueError("Only augment integer encoded scene")

        # 1. Flip the scene horizontally
        flipped_scene = torch.flip(scene.clone(), dims=[-1])

        # 2. Swap tile types 1 and 2 (tops of pipes)
        mask_1 = (flipped_scene == 1)
        mask_2 = (flipped_scene == 2)
        # Swap values using masks
        flipped_scene[mask_1] = 2
        flipped_scene[mask_2] = 1

        # 3. Swap tile types 9 and 10 (bodies of pipes)
        mask_9 = (flipped_scene == 9)
        mask_10 = (flipped_scene == 10)
        # Swap values using masks
        flipped_scene[mask_9] = 10
        flipped_scene[mask_10] = 9

        # Change left to right and vice versar
        caption_tensor = torch.tensor(self._swap_caption_tokens(caption), dtype=torch.long)

        return flipped_scene, caption_tensor

    def _shuffle_data(self):
        """Shuffles the dataset."""
        combined = list(zip(self.data, self.tokenized_captions))
        random.shuffle(combined)
        self.data, self.tokenized_captions = zip(*combined)

    def _swap_caption_tokens(self, caption_tensor):
        """swapping directional tokens for consistency with flipped scenes"""

        # If locations are not in captions, then left/right will not exist
        left_id = self.tokenizer.token_to_id["left"] if "left" in self.tokenizer.token_to_id else -1
        right_id = self.tokenizer.token_to_id["right"] if "right" in self.tokenizer.token_to_id else -1
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

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.tokenized_captions)

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
        caption_tokens = self.tokenizer.pad_sequence(caption_tokens, self.max_length)
        caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)

        if self.mode == "mlm":
            return caption_tensor  # MLM only uses captions

        scene_tensor = torch.tensor(sample["scene"], dtype=torch.long)  # Convert scene to tensor
        
        # Apply augmentation if enabled
        if self.augment and random.choice([True, False]):
            #print("AUGMENT!", idx)
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
    
        Returns:
            List of lists of integers representing the original scene layout
        """
        # Check if we have a batched input
        is_batched = len(one_hot_scene.shape) == 4
    
        if is_batched:
            print(one_hot_scene.shape)
            raise ValueError("Call decode_scene with a single scene, not a batch")
    
        # Permute back to [height, width, num_tiles] format
        one_hot_permuted = one_hot_scene.permute(1, 2, 0)
    
        # Get the indices (tile IDs) where the one-hot encoding has a 1
        scene_indices = torch.argmax(one_hot_permuted, dim=2)
    
        # Convert to a list of lists
        scene_list = scene_indices.tolist()
    
        return scene_list

if __name__ == "__main__":

    random.seed(0)

    tokenizer = Tokenizer()
    tokenizer.load('SMB1_Tokenizer.pkl')

    # Create MLM dataset
    mlm_dataset = LevelDataset('SMB1_LevelsAndCaptions.json', tokenizer, mode="mlm")
    sample = mlm_dataset[0]
    print("MLM sample shape:", sample.shape)  # Should be (max_length)
    print(sample)
    print(mlm_dataset.tokenizer.decode(sample.tolist()))

    mlm_dataloader = DataLoader(mlm_dataset, batch_size=16, shuffle=True)
    batch = next(iter(mlm_dataloader))
    print("MLM batch shape:", batch.shape)  # Should be (16, max_length)
    print(batch[0])
    print(mlm_dataset.tokenizer.decode(batch[0].tolist()))

    # Create Diffusion dataset
    diffusion_dataset = LevelDataset('SMB1_LevelsAndCaptions.json', tokenizer, mode="diffusion", shuffle=False)
    scene, caption = diffusion_dataset[0]
    print("Diffusion Sample Shapes:", scene.shape, caption.shape) 
    print(scene)
    print(torch.tensor(diffusion_dataset.decode_scene(scene)))
    print(diffusion_dataset.tokenizer.decode(caption.tolist()))

    diffusion_dataloader = DataLoader(diffusion_dataset, batch_size=16, shuffle=False)
    scenes, captions = next(iter(diffusion_dataloader))
    print("Diffusion Batch Shapes:", scenes.shape, captions.shape) 

    print(scenes[0])
    print(torch.tensor(diffusion_dataset.decode_scene(scenes[0])))
    print(diffusion_dataset.tokenizer.decode(captions[0].tolist()))

    print("-----------")

    diffusion_dataset.augment = False
    scene, caption = diffusion_dataset[290]
    print(torch.tensor(diffusion_dataset.decode_scene(scene)))
    print(diffusion_dataset.tokenizer.decode(caption.tolist()))
    diffusion_dataset.augment = True # Augmentation is random, so won't always be different
    scene, caption = diffusion_dataset[290]
    print(torch.tensor(diffusion_dataset.decode_scene(scene)))
    print(diffusion_dataset.tokenizer.decode(caption.tolist()))

    print("-----------")
    itr = iter(diffusion_dataloader)
    for i in range(17): next(itr) # Skip batches
    # batch is (scenes, captions) so the [0] gets just the scenes
    visualize_samples(next(itr)[0], "TEMP")