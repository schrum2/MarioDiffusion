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
from captions.caption_match import TOPIC_KEYWORDS, BROKEN_TOPICS, KEYWORD_TO_NEGATED_PLURAL
import numpy as np

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

def convert_to_level_format(sample, block_embeddings=None):
    """
    Convert model output to level indices
    Expected input shape: [samples, channels, height, width]
    """
    if block_embeddings:
        # Reshape sample to [batch_size * height * width, embedding_dim]
        batch_size, embedding_dim, height, width = sample.shape
        flat_samples = sample.permute(0, 2, 3, 1).reshape(-1, embedding_dim)
        
        # Normalize vectors for cosine similarity
        flat_samples = F.normalize(flat_samples, p=2, dim=1)
        block_embeddings = F.normalize(block_embeddings, p=2, dim=1)
        
        # Calculate cosine similarity between each position and all tile embeddings
        similarities = torch.matmul(flat_samples, block_embeddings.t())
        
        # Get indices of most similar tiles
        indices = torch.argmax(similarities, dim=1)
        
        # Reshape back to [batch_size, height, width]
        indices = indices.reshape(batch_size, height, width)
        
        return indices.cpu().numpy()

        # #use cosine similarity to get the closest tile
        # # go through samples
        # print(sample.shape)
        # quit()
        # return None
    else:
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

def visualize_samples(samples, output_dir=None, use_tiles=True, start_index=0):
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
    #if samples.shape[1] != 15:
        #print(samples.shape)
        #raise ValueError("Hard coded for 15 channels (change code to generalize beyond Mario)")

    # Create directory for the samples
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert from one-hot to tile indices
    # sample_indices = []
    sample_indices = convert_to_level_format(samples)
    num_samples = len(samples)
    grid_cols = min(4, num_samples)  # Limit to 4 columns
    grid_rows = (num_samples + grid_cols - 1) // grid_cols  # Calculate rows needed

    if use_tiles:
        tile_images = tiles()
        tile_size = 16 # Specifically for Mario

        for i, sample_index in enumerate(sample_indices):
            # Create a blank image to hold the tile-based visualization
            height, width = sample_index.shape
            composite_image = Image.new('RGB', (width * tile_size, height * tile_size))

            for row in range(height):
                for col in range(width):
                    tile_id = int(sample_index[row, col] % len(tile_images))  # Ensure tile_id is within bounds
                    tile_image = tile_images[tile_id]
                    composite_image.paste(tile_image, (col * tile_size, row * tile_size))

            if output_dir:
                composite_image.save(os.path.join(output_dir, f"sample_{i + start_index}.png"))
            else:
                return composite_image

    else:
        # Create custom colormap for integers 0-15
        colorslist = colors()
        custom_cmap = matplotlib.colors.ListedColormap(colorslist[:15])

        plt.figure(figsize=(4 * grid_cols, 4 * grid_rows))  # Adjust figure size dynamically

        for i, sample_index in enumerate(sample_indices):

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

def positive_negative_caption_split(caption, remove_upside_down_pipes, randomize=False):
    phrases = [p.strip() for p in caption.split(".") if p]
    positive_phrases = ""
    negative_phrases = ""

    if "no " not in caption and len(phrases) == len(TOPIC_KEYWORDS) - BROKEN_TOPICS:
        positive_phrases = caption
    elif "no " in caption and len(phrases) == len(TOPIC_KEYWORDS) - BROKEN_TOPICS:
        positive_phrases = ". ".join([p for p in phrases if "no " not in p]) + "."
        negative_phrases = ". ".join([p.replace("no ", "") for p in phrases if "no " in p]) + "."
    elif "no " in caption:
        raise ValueError(f"With negative phrases, every topic should be represented: {caption} {len(phrases)} {len(TOPIC_KEYWORDS)} {TOPIC_KEYWORDS}")
    elif len(phrases) < len(TOPIC_KEYWORDS) - BROKEN_TOPICS:
        positive_phrases = caption
        negative_phrases = ". ".join([f"{topic}" for topic in (random.sample(TOPIC_KEYWORDS, len(TOPIC_KEYWORDS)) if randomize else TOPIC_KEYWORDS) if topic not in caption]) + "."
        for src, target in KEYWORD_TO_NEGATED_PLURAL:
            negative_phrases = negative_phrases.replace(src, target)
    else:
        raise ValueError(f"Caption has problem: {caption} {len(phrases)} {len(TOPIC_KEYWORDS)}")

    if remove_upside_down_pipes:
        # Remove upside down pipes from negative phrases
        negative_phrases = negative_phrases.replace(" upside down pipes.", "")
        negative_phrases = negative_phrases.replace("upside down pipes. ", "")

    return positive_phrases, negative_phrases

class LevelDataset(Dataset):
    def __init__(self, json_path, tokenizer, shuffle=True, max_length=None, mode="diffusion", augment=True, random_flip=False, limit=-1, num_tiles=15, negative_captions=False, block_embeddings=None):
        """
            Args:
            json_path (str): Path to JSON file with captions.
            tokenizer (Tokenizer): Tokenizer instance.
            shuffle (bool): Whether to shuffle data at the start of an epoch.
            max_length (int, optional): Maximum length for tokenized captions.
            mode (str): "diffusion" for level scenes + captions tokens, 
                        "mlm" for masked language model training (tokenized captions only), 
                        "text" for just the text captions, 
                        "diff_text" for level scenes and text captions (used with a pretrained model).
            augment (bool): Whether to apply data augmentation to text captions.
            random_flip (bool): Whether to randomly flip the scene and caption.
            limit (int): restrict dataset to this size if not -1
            num_tiles (int): Number of different tile types for one-hot encoding
        """
        assert mode in ["mlm", "diffusion","text", "diff_text"], "Mode must be 'mlm', 'text', 'diffusion', or 'diff_text'."

        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.augment = augment
        self.random_flip = random_flip
        self.num_tiles = num_tiles
        self.negative_captions = negative_captions

        # For embeddings
        self.block_embeddings = block_embeddings # Store block embeddings
        # self.levels = self.load_levels()
        # self.tile_counts = self.calculate_tile_counts()
        # self.level_lengths = [len(level) for level in self.levels]
        # self.cumulative_lengths = np.cumsum(self.level_lengths)
        # self.max_len = max(self.level_lengths)
        self.pad_token = tokenizer.token_to_id["[PAD]"] if tokenizer else None
        # self.augmentor = LevelAugmentor()
        # self.data = self.process_data()

        # Load data
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        if limit > -1:
            # Random selection of limited portion of data (if limit is less than actual size)
            self.data = random.sample(self.data, limit)

        print(f"Training samples: {len(self.data)}")

        # Determine padding length (if not provided)
        if self.max_length is None:
            # Add 5 just in case
            self.max_length = max(len(caption.replace(".", " .").split()) for caption in (item["caption"] for item in self.data)) + 5

        # Shuffle dataset
        if self.shuffle:
            random.shuffle(self.data)

        remove_upside_down_pipes = False
        if self.negative_captions:
            # If the captions do not contain upside down pipes, then the negative captions
            # should never say there are no upside down pipes too.
            remove_upside_down_pipes = True
            for sample in self.data:
                caption = sample["caption"]
                if "upside" in caption:
                    # No problem. Upside down pipes are present
                    remove_upside_down_pipes = False
                    break

        self.remove_upside_down_pipes = remove_upside_down_pipes
        print("remove_upside_down_pipes:", self.remove_upside_down_pipes)

    def _augment_caption(self, caption):
        """Shuffles period-separated phrases in the caption."""
        if self.augment:
            phrases = caption[:-1].split(". ") # [:-1] removes the last period
            random.shuffle(phrases)  # Shuffle phrases
            return ". ".join(phrases) + "."
        else:
            return caption # Same as original

    def _flip_scene(self, scene): # augments by flipping
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

        return flipped_scene

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
        return len(self.data)

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

        negative_caption = ""
        if self.negative_captions:
            augmented_caption, negative_caption = positive_negative_caption_split(augmented_caption, self.remove_upside_down_pipes, self.augment)

        if self.mode == "text":
            if self.negative_captions:
                # Return the raw caption for text mode
                return augmented_caption, negative_caption
            else:
                # Return the raw caption for text mode
                return augmented_caption

        if self.mode != "diff_text":
            caption_tokens = self.tokenizer.encode(augmented_caption)
            if len(caption_tokens) > self.max_length:
                raise ValueError(f"Caption length exceeds max_length: {len(caption_tokens)} > {self.max_length}: {augmented_caption}")
            
            caption_tokens = self.tokenizer.pad_sequence(caption_tokens, self.max_length)
            caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)

            if self.negative_captions:
                negative_caption_tokens = self.tokenizer.encode(negative_caption)
                negative_caption_tokens = self.tokenizer.pad_sequence(negative_caption_tokens, self.max_length)
                negative_caption_tensor = torch.tensor(negative_caption_tokens, dtype=torch.long)

            if self.mode == "mlm":
                if self.negative_captions:
                    return caption_tensor, negative_caption_tensor
                else:
                    return caption_tensor  # MLM only uses caption tokens

        scene_tensor = torch.tensor(sample["scene"], dtype=torch.long)  # Convert scene to tensor
        
        # Apply random flip if enabled
        if self.random_flip and random.choice([True, False]):
            scene_tensor = self._flip_scene(scene_tensor)
            if self.mode != "diff_text":
                caption_tensor = torch.tensor(self._swap_caption_tokens(caption), dtype=torch.long)


        # Added to support embeddings
        if self.block_embeddings is not None:
            #raise ValueError("Block embeddings not supported yet")
            # Replace one-hot encoding with block embeddings
            one_hot_scene = torch.stack([self.block_embeddings[tile_id] for tile_id in scene_tensor])
        else:
            one_hot_scene = F.one_hot(scene_tensor, num_classes=self.num_tiles).float()
            # Permute dimensions to [num_tiles, height, width]
            #print("before permute", one_hot_scene.shape)
            # one_hot_scene = one_hot_scene.permute(2, 0, 1)
            #print("after permute", one_hot_scene.shape)

        one_hot_scene = one_hot_scene.permute(2, 0, 1)

        if self.mode == "diff_text":
            # Return the raw caption for the pretrained model, this should be moved up later
            #TODO: add support for negative captions
            return one_hot_scene, augmented_caption
        
        # The options below include the scene, but also the tokenized captions

        if self.negative_captions:
            return one_hot_scene, caption_tensor, negative_caption_tensor
        else:
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

        # Change so this uses convert_to_level_format

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
    torch.manual_seed(0)  # Add PyTorch seed for DataLoader determinism

    tokenizer = Tokenizer()
    tokenizer.load('SMB1AND2_Tokenizer-absence.pkl')

    # Create Diffusion dataset
    diffusion_dataset = LevelDataset('SMB1_LevelsAndCaptions-regular.json', tokenizer, mode="diffusion", shuffle=False, block_embeddings=None) 
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


    quit()













    negatives_mlm_dataset = LevelDataset('SMB1AND2_LevelsAndCaptions-absence.json', tokenizer, mode="text", negative_captions=True)
    print("Negative MLM dataset size:", len(negatives_mlm_dataset))
    for i in range(5):
        sample = negatives_mlm_dataset[i]
        print(i)
        print(f"      POS: {sample[0]}")
        print(f"      NEG: {sample[1]}")

    print("----------------------------------")

    tokenizer = Tokenizer()
    tokenizer.load('SMB1AND2_Tokenizer-regular.pkl')

    negatives_mlm_dataset = LevelDataset('SMB1AND2_LevelsAndCaptions-regular.json', tokenizer, mode="text", negative_captions=True)
    print("Negative MLM dataset size:", len(negatives_mlm_dataset))
    for i in range(5):
        sample = negatives_mlm_dataset[i]
        print(i)
        print(f"      POS: {sample[0]}")
        print(f"      NEG: {sample[1]}")

    print("----------------------------------")


    tokenizer = Tokenizer()
    tokenizer.load('SMB1AND2_Tokenizer-absence.pkl')

    negatives_mlm_dataset = LevelDataset('SMB1AND2_LevelsAndCaptions-absence.json', tokenizer, mode="mlm", negative_captions=True)
    print("Negative MLM dataset size:", len(negatives_mlm_dataset))
    for i in range(5):
        sample = negatives_mlm_dataset[i]
        print(i)
        print(f"      POS: {sample[0]}")
        print(f"      POS: {tokenizer.decode(sample[0].tolist())}")
        print(f"      NEG: {sample[1]}")
        print(f"      NEG: {tokenizer.decode(sample[1].tolist())}")

    print("----------------------------------")

    tokenizer = Tokenizer()
    tokenizer.load('SMB1AND2_Tokenizer-regular.pkl')

    negatives_mlm_dataset = LevelDataset('SMB1AND2_LevelsAndCaptions-regular.json', tokenizer, mode="mlm", negative_captions=True)
    print("Negative MLM dataset size:", len(negatives_mlm_dataset))
    for i in range(5):
        sample = negatives_mlm_dataset[i]
        print(i)
        print(f"      POS: {sample[0]}")
        print(f"      POS: {tokenizer.decode(sample[0].tolist())}")
        print(f"      NEG: {sample[1]}")
        print(f"      NEG: {tokenizer.decode(sample[1].tolist())}")

    print("----------------------------------")


    tokenizer = Tokenizer()
    tokenizer.load('SMB1AND2_Tokenizer-absence.pkl')

    negatives_mlm_dataset = LevelDataset('SMB1AND2_LevelsAndCaptions-absence.json', tokenizer, mode="diffusion", negative_captions=True)
    print("Negative MLM dataset size:", len(negatives_mlm_dataset))
    for i in range(5):
        sample = negatives_mlm_dataset[i]
        print(i)
        print(f"      POS: {sample[1]}")
        print(f"      POS: {tokenizer.decode(sample[1].tolist())}")
        print(f"      NEG: {sample[2]}")
        print(f"      NEG: {tokenizer.decode(sample[2].tolist())}")

    print("----------------------------------")

    tokenizer = Tokenizer()
    tokenizer.load('SMB1AND2_Tokenizer-regular.pkl')

    negatives_mlm_dataset = LevelDataset('SMB1AND2_LevelsAndCaptions-regular.json', tokenizer, mode="diffusion", negative_captions=True)
    print("Negative MLM dataset size:", len(negatives_mlm_dataset))
    for i in range(5):
        sample = negatives_mlm_dataset[i]
        print(i)
        print(f"      POS: {sample[1]}")
        print(f"      POS: {tokenizer.decode(sample[1].tolist())}")
        print(f"      NEG: {sample[2]}")
        print(f"      NEG: {tokenizer.decode(sample[2].tolist())}")

    print("----------------------------------")

    # Create MLM dataset
    mlm_dataset = LevelDataset('Mario_LevelsAndCaptions.json', tokenizer, mode="mlm")
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
    diffusion_dataset = LevelDataset('Mario_LevelsAndCaptions.json', tokenizer, mode="diffusion", shuffle=False)
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

    print("-----------")
    tokenizer = Tokenizer()
    tokenizer.load('Mario_Tokenizer.pkl')
    mlm_dataset = LevelDataset('Mario_LevelsAndCaptions.json', tokenizer, mode="mlm")
    last_size = None
    for b in mlm_dataset:
        if last_size == None:
            print(b.shape)
            last_size = b.shape
        elif last_size != b.shape:
            print("Different!")
            print(b.shape)
            print(b)
            break
            last_size = b.shape
        