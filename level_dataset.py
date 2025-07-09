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
import util.common_settings as common_settings
import re

# Global variable to store the loaded sprite sheet
_sprite_sheet = None
_sprite_sheet_name = None

def samples_to_scenes(all_samples, block_embeddings=None):
    # Convert to list
    samples_list = [all_samples[i] for i in range(len(all_samples))]
    scenes = []
    # Process and collect individual samples
    for _, sample in enumerate(samples_list):
        # Convert to indices
        sample_tensor = sample.unsqueeze(0) # if sample.shape[0] == args.num_tiles else sample
        sample_indices = convert_to_level_format(sample_tensor, block_embeddings)
        
        # Add level data to the list
        scene = sample_indices[0].tolist() # Always just one scene: (1,16,16)
        scenes.append(scene)

    return scenes

def convert_to_level_format(sample, block_embeddings=None):
    """
    Convert model output to level indices
    Expected input shape: [samples, channels, height, width]
    """
    if block_embeddings is not None:
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
    # Create custom colormap for integers 
    colorslist = [
        (0.2, 0.3, 0.7),    # 0 = darker blue: sky
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
        (0.3, 0.3, 0.3),    # 15 (extra color just in case)
        (0.72, 0.031, 0.753),
        (0.912, 0.215, 0.708),
        (0.82, 0.672, 0.166),
        (0.342, 0.25, 0.571),
        (0.013, 0.055, 0.842),
        (0.078, 0.938, 0.688),
        (0.172, 0.056, 0.087),
        (0.062, 0.608, 0.968), #Extra randomly-generated colors for long mega man data
        (0.001, 0.478, 0.136),
        (0.542, 0.81, 0.345),
        (0.541, 0.478, 0.703),
        (0.596, 0.108, 0.466),
        (0.27, 0.453, 0.655),
        (0.187, 0.037, 0.295),
        (0.783, 0.744, 0.474),
        (0.333, 0.036, 0.349),
        (0.491, 0.736, 0.145),
        (0.362, 0.128, 0.78),
        (0.401, 0.028, 0.866),
        (0.486, 0.748, 0.975),
        (0.787, 0.462, 0.722),
        (0.694, 0.804, 0.86),
        (0.647, 0.801, 0.301)
    ]

    return colorslist

def mario_tiles():
    """
    Maps integers 0-15 to 16x16 pixel sprites from mapsheet.png.

    Returns:
        A list of 16x16 pixel tile images for Mario.
    """

    # DEBUGGING
    #raise ValueError("Why is this being called!")

    global _sprite_sheet
    global _sprite_sheet_name


    # Load the sprite sheet only once
    if _sprite_sheet_name != "mapsheet.png":
        _sprite_sheet_name = "mapsheet.png" #Done to ensure we can change the sprite sheet after code execution
        _sprite_sheet = Image.open(_sprite_sheet_name)

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
        # Tile right below decides what the padded tile is (sky currently)
        (2,5),    # 13 = Padding (sky)
        (0,6),    # 14 = Nothing
        (1,6),    # 15 = Nothing (extra just in case)
    ]

    # Extract each tile as a 16x16 image
    tile_images = []
    for col, row in tile_coordinates:
        left = col * common_settings.MARIO_TILE_PIXEL_DIM
        upper = row * common_settings.MARIO_TILE_PIXEL_DIM
        right = left + common_settings.MARIO_TILE_PIXEL_DIM
        lower = upper + common_settings.MARIO_TILE_PIXEL_DIM
        tile = _sprite_sheet.crop((left, upper, right, lower))
        tile_images.append(tile)

    # Add a blank tile for the extra tile (padding)
    blank_tile = Image.new('RGB', (common_settings.MARIO_TILE_PIXEL_DIM, common_settings.MARIO_TILE_PIXEL_DIM), color=(128, 128, 128))  # Gray or any color
    tile_images.append(blank_tile)

    # Save each tile image as tile_X.png for inspection
    #for idx, tile_img in enumerate(tile_images):
    #    tile_img.save(f"tile_{idx}.png")

    return tile_images

def lr_tiles():
    """
    Maps integers 0-10 to 8x8 pixel sprites from LR_mapsheet.png.

    Returns:
        A list of 8x8 pixel tile images for Lode Runner.
    """
    global _sprite_sheet
    global _sprite_sheet_name

    # Load the sprite sheet only once
    if _sprite_sheet_name != "LR_mapsheet.png":
        _sprite_sheet_name = "LR_mapsheet.png" #Done to ensure we can change the sprite sheet after code execution
        _sprite_sheet = Image.open(_sprite_sheet_name)

    # Hardcoded coordinates for the first 10 tiles (row, col)
    LR_tile_coordinates = [
       (12, 4),     # 0 = Ladder            done
       (14, 4),     # 1 = Rope              done
       (1, 1),      # 2 = Passable, Empty   done
       (2, 3),      # 3 = Solid Ground      done
       (3, 2),      # 4 = Enemy             done
       (5, 2),      # 5 = Gold              done
       (18, 21),    # 6 = Spawn             done
       (1, 22),     # 7 = Diggable Ground   done
       # Tile right below decides what the padded tile is (empty space currently)
       (1,1)        # 8 = Padding           done

    ]

    DIM = 8

    # Extract each tile as a 8x8 image
    LR_tile_images = []
    for col, row in LR_tile_coordinates:
        left = col * DIM
        upper = 4 + row * DIM
        right = left + DIM
        lower = upper + DIM
        tile = _sprite_sheet.crop((left, upper, right, lower))
        LR_tile_images.append(tile)

    # Add a blank tile for the extra tile (padding)
    blank_tile = Image.new('RGB', (DIM, DIM), color=(128, 128, 128))
    LR_tile_images.append(blank_tile)

    return LR_tile_images


def mm_tiles(game):
    """
    Maps integers 0-11 or 0-38 to 16x16 pixel sprites from MM_mapsheet.png.

    Returns:
        A list of 16x16 pixel tile images for Mega Man.
    """
    global _sprite_sheet
    global _sprite_sheet_name

    # Load the sprite sheet only once
    if _sprite_sheet_name != "MM_mapsheet.png":
        _sprite_sheet_name = "MM_mapsheet.png" #Done to ensure we can change the sprite sheet after code execution
        _sprite_sheet = Image.open(_sprite_sheet_name)

    # Hardcoded coordinates for the first 10 tiles (row, col)
    if game == 'MM-Full':
        MM_tile_coordinates = [
            (0,0),    #0 = Player/Spawn point
            (0,1),    #1 = null
            (0,2),    #2 = air/empty tile
            (0,3),    #3 = Water
            (0,4),    #4 = ground/wall
            (0,5),    #5 = Ladder
            (0,6),    #6 = Breakable block
            (0,7),    #7 = Fake blocks (look solid but aren't)
            (0,8),    #8 = Appearing/disappearing block
            (0,9),    #9 = Moving platform
            (0,10),   #10 = Door

            (1,0),    #11 = Large ammo pack
            (1,1),    #12 = Small ammo pack
            (1,2),    #13 = Large health pack
            (1,3),    #14 = Small health pack
            (1,4),    #15 = Extra life
            (1,5),    #16 = Yashichi, a special item that completely fills health and ammo (only shows up in the final level)
            (1,6),    #17 = Magnet Beam (one-time appearance)
            (1,7),    #18 = Orb collectable to get a new weapon

            (2,0),    #19 = Spikes
            (2,1),    #20 = Fire Pillar

            (3,0),    #21 = Foot holder enemy/platform
            (3,1),    #22 = Sniper Joe enemy
            (3,2),    #23 = Flea enemy
            (3,3),    #24 = Flying shell enemy spawner
            (3,4),    #25 = Killer bullet enemy spawner
            (3,5),    #26 = Killer bullet enemy
            (3,6),    #27 = Spine enemy
            (3,7),    #28 = Beak enemy
            (3,8),    #29 = Screw bomber enemy
            (3,9),    #30 = Tackle fire enemy
            (3,10),   #31 = Watcher enemy

            (4,0),    #32 = Octopus battery enemy going up/down
            (4,1),    #33 = Octopus battery enemy going left/right
            (4,2),    #34 = Big eye enemy
            (4,3),    #35 = Bunby Heli enemy
            (4,4),    #36 = Met enemy
            (4,5),    #37 = Picket man enemy
            (4,6)     #38 = Crazy razy enemy
        ]
    else:
        MM_tile_coordinates = [
            (0,4),     #0 = ground/wall
            (0,1),     #1 = null
            (0,8),     #2 = Appearing/disappearing block
            (0,6),     #3 = Breakable block
            (2,1),     #4 = Fire Pillar
            (0,10),    #5 = Door
            (2,0),     #6 = Spikes
            (0,9),     #7 = Moving platform
            (0,5),     #8 = Ladder
            (0,3),     #9 = Water
            (4,4),     #10 = Met enemy
            (1,3),     #11 = Small health pack
            (0,2)      #12 = air/empty tile
        ]

    DIM = common_settings.MM_TILE_PIXEL_DIM

    # Extract each tile as a 16x16 image
    MM_tile_images = []
    for row, col in MM_tile_coordinates:
        left = col * DIM
        upper = row * DIM
        right = left + DIM
        lower = upper + DIM
        tile = _sprite_sheet.crop((left, upper, right, lower))
        MM_tile_images.append(tile)

    # Add a blank tile for the extra tile (padding)
    blank_tile = Image.new('RGB', (DIM, DIM), color=(128, 128, 128))
    MM_tile_images.append(blank_tile)

    return MM_tile_images



def visualize_samples(samples, output_dir=None, use_tiles=True, start_index=0, block_embeddings=None, prompts=None, game='Mario'):
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

    # Create directory for the samples
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert from one-hot to tile indices
    # sample_indices = []
    sample_indices = convert_to_level_format(samples, block_embeddings)
    #print(sample_indices.shape)
    num_samples = len(samples)
    grid_cols = min(4, num_samples)  # Limit to 4 columns
    grid_rows = (num_samples + grid_cols - 1) // grid_cols  # Calculate rows needed

    if use_tiles:
        channels = samples.shape[1]
        height = samples.shape[2]
        width = samples.shape[3]
        if game == 'LR': #and width == lr_common_settings.LR_WIDTH:
            #print("Using Lode Runner tiles")
            tile_images = lr_tiles()
            tile_size = common_settings.LR_TILE_PIXEL_DIM
        #elif height == common_settings.MARIO_HEIGHT: #and width == common_settings.MARIO_WIDTH:
        elif game == 'Mario': # Default to Mario
            #print("Using Mario tiles")
            tile_images = mario_tiles()
            tile_size = common_settings.MARIO_TILE_PIXEL_DIM
        elif game == 'MM-Simple' or game == 'MM-Full':
            tile_images = mm_tiles(game)
            tile_size = common_settings.MM_TILE_PIXEL_DIM
        else:
            raise ValueError(f"Unsupported game or dimensions: {game} {height}x{width}")
        
        for i, sample_index in enumerate(sample_indices):
            # Create a blank image to hold the tile-based visualization
            height, width = sample_index.shape
            composite_image = Image.new('RGB', (width * tile_size, height * tile_size))

            for row in range(height):
                for col in range(width):
                    tile_id = int(sample_index[row, col] % len(tile_images))  # Ensure tile_id is within bounds
                    tile_image = tile_images[tile_id]
                    composite_image.paste(tile_image, (col * tile_size, row * tile_size))

            # Determine the file name based on the prompt
            if prompts:
                sanitized_prompt = prompts[i].replace(".", "")[:50]
                file_name = f"sample_{i + start_index} - {sanitized_prompt}.png"
            else:
                file_name = f"sample_{i + start_index} - unconditional.png"

            if output_dir:
                composite_image.save(os.path.join(output_dir, file_name))
            else:
                return composite_image

    else:
        # Create custom colormap for integers 
        colorslist = colors()
        custom_cmap = matplotlib.colors.ListedColormap(colorslist[:15])

        fig = plt.figure(figsize=(4 * grid_cols, 4 * grid_rows))  # Adjust figure size dynamically
        try:
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
        finally:
            # Always close the figure, even if an error occurs
            plt.close(fig)

        # Returning an image instead of saving many images
        if result:
            return result

    return sample_indices

def remove_duplicate_phrases(text):
    seen = set()
    topic_to_phrase = {}
    
    # Normalize phrases
    raw_phrases = [p.strip() for p in text.split('.') if p.strip()]

    for phrase in raw_phrases:
        phrase_lower = phrase.lower()
        if phrase_lower in seen:
            continue  # Skip exact duplicates
        seen.add(phrase_lower)

        # Find topic keyword match
        matched_topic = None
        for topic in TOPIC_KEYWORDS:
            if topic in phrase_lower:
                matched_topic = topic
                break
        
        if matched_topic:
            existing = topic_to_phrase.get(matched_topic)
            # Prefer positive version over 'no' version
            if existing:
                if existing.lower().startswith("no ") and not phrase_lower.startswith("no "):
                    topic_to_phrase[matched_topic] = phrase  # Replace 'no X' with positive
                # else keep existing
            else:
                topic_to_phrase[matched_topic] = phrase
        else:
            # Not in any topic: keep as-is
            topic_to_phrase[phrase] = phrase

    return '. '.join(topic_to_phrase.values()) + '.'
    
def append_absence_captions(prompt, topic_keywords=TOPIC_KEYWORDS):
    """
    Appends 'no X' for each topic in topic_keywords not mentioned in the prompt.
    Avoids false positives for substrings (e.g., 'pipe' vs 'upside down pipe').
    Skips adding absence phrases for topics containing the word 'broken'.
    """
    import re

    prompt_lower = prompt.lower()
    phrases = [p.strip() for p in re.split(r'[.,;]', prompt_lower) if p.strip()]
    absence_phrases = []

    # Build a set of topics that are present in the prompt
    present_topics = set()
    for topic in topic_keywords:
        for phrase in phrases:
            if any(word.startswith(topic) for word in phrase.split()):
                present_topics.add(topic)
                break

    # For each topic, if not present, add "no X" unless 'broken' is in the topic
    for topic in topic_keywords:
        if topic in present_topics:
            continue
        if 'broken' in topic.lower():
            # print(f"[Skip] Topic contains 'broken': {topic}")
            continue
        elif topic in {"rectangular", "irregular"}:
            absence_phrases.append(f"no {topic} block clusters")
        elif topic in {"ceiling", "floor"}:
            absence_phrases.append(f"no {topic}")
        elif topic == "enem":
            absence_phrases.append("no enemies")
        elif topic not in {"ceiling", "floor"}:
            absence_phrases.append(f"no {topic}s")
        

    if absence_phrases:
        result = prompt.rstrip(" .") + ". " + ". ".join(absence_phrases) + "."
        return result
    else:
        return prompt




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
    def __init__(self, json_path=None, tokenizer=None, data_as_list=None, shuffle=True, max_length=None, mode="diff_text", augment=True, random_flip=False, limit=-1, num_tiles=common_settings.MARIO_TILE_COUNT, negative_captions=False, block_embeddings=None):
        """
            Args:
            json_path (str): Path to JSON file with captions.
            tokenizer (Tokenizer): Tokenizer instance.
            shuffle (bool): Whether to shuffle data at the start of an epoch.
            max_length (int, optional): Maximum length for tokenized captions.
            mode (str): "text" for just the text captions, 
                        "diff_text" for level scenes and text captions (used with a pretrained model).
            augment (bool): Whether to apply data augmentation to text captions.
            random_flip (bool): Whether to randomly flip the scene and caption.
            limit (int): restrict dataset to this size if not -1
            num_tiles (int): Number of different tile types for one-hot encoding
        """
        assert mode in ["text", "diff_text"], "Mode must be 'text' or 'diff_text'."

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
        if json_path is None and data_as_list:
            print(f"Data given as list")
            self.data = data_as_list  
        elif not os.path.exists(json_path):
            raise ValueError(f"JSON file does not exist: {json_path}")
        else:
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


    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches one sample.

        Returns:
            - In "text" mode: raw augmented caption (string) 
                or a tuple of positive and negative captions if negative_captions is True.
            - In "diff_text" mode: (scene_tensor, augmented_caption)
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

        scene_tensor = torch.tensor(sample["scene"], dtype=torch.long)  # Convert scene to tensor
        
        # Apply random flip if enabled
        if self.random_flip and random.choice([True, False]):
            scene_tensor = self._flip_scene(scene_tensor)

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

        if self.negative_captions:
            return one_hot_scene, augmented_caption, negative_caption
        else:
            return one_hot_scene, augmented_caption

        

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
        if len(one_hot_scene.shape) == 4:
            raise ValueError("Call decode_scene with a single scene, not a batch")
        
        # Add batch dimension for convert_to_level_format
        scene = one_hot_scene.unsqueeze(0)  # [1, channels, height, width]

        # Use convert_to_level_format with appropriate block embeddings
        indices = convert_to_level_format(scene, self.block_embeddings)

        # Remove batch dimension and convert to list
        scene_list = indices[0].tolist()  # [height, width]
        return scene_list

        # # Check if we have a batched input
        # is_batched = len(one_hot_scene.shape) == 4
    
        # if is_batched:
        #     print(one_hot_scene.shape)
        #     raise ValueError("Call decode_scene with a single scene, not a batch")
    
        # # Permute back to [height, width, num_tiles] format
        # one_hot_permuted = one_hot_scene.permute(1, 2, 0)
    
        # # Get the indices (tile IDs) where the one-hot encoding has a 1
        # scene_indices = torch.argmax(one_hot_permuted, dim=2)
    
        # # Convert to a list of lists
        # scene_list = scene_indices.tolist()
    
        # return scene_list






from models.block2vec_model import Block2Vec

if __name__ == "__main__":

    random.seed(0)
    torch.manual_seed(0)  # Add PyTorch seed for DataLoader determinism

    tokenizer = Tokenizer()
    tokenizer.load('datasets\Mar1and2_Tokenizer-regular.pkl')

    # Load block embeddings
    block2vec = Block2Vec.from_pretrained("SMB1-block2vec-embeddings")
    block_embeddings = block2vec.get_embeddings()
    # Create Diffusion dataset
    diffusion_dataset = LevelDataset(
        'datasets/Mar1and2_LevelsAndCaptions-regular.json',
        tokenizer, 
        mode="diff_text", 
        shuffle=False,
        block_embeddings=block_embeddings
    )

    for i, emb in enumerate(block_embeddings):
        print(f"Tile {i}: {emb}")

    scene, caption = diffusion_dataset[0]
    print(caption)
    print("Diffusion Sample Shapes:", scene.shape, caption) 
    print(scene)
    print(torch.tensor(diffusion_dataset.decode_scene(scene)))
    print(caption)

    diffusion_dataloader = DataLoader(diffusion_dataset, batch_size=16, shuffle=False)
    scenes, captions = next(iter(diffusion_dataloader))
    print("Diffusion Batch Shapes:", scenes.shape, captions) 

    print(f"raw scene: {scenes[10]}")
    print(f"proccesed scene: {torch.tensor(diffusion_dataset.decode_scene(scenes[10]))}")
    print(captions[10])

    print(scenes.shape)
    image = visualize_samples(scenes, output_dir="TEMP", use_tiles=True, start_index=0, block_embeddings=block_embeddings)


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

    negatives_mlm_dataset = LevelDataset('SMB1AND2_LevelsAndCaptions-absence.json', tokenizer, mode="diff_text", negative_captions=True)
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

    negatives_mlm_dataset = LevelDataset('SMB1AND2_LevelsAndCaptions-regular.json', tokenizer, mode="diff_text", negative_captions=True)
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
    diffusion_dataset = LevelDataset('Mario_LevelsAndCaptions.json', tokenizer, mode="diff_text", shuffle=False)
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
