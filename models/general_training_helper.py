from torch.utils.data import DataLoader
from level_dataset import LevelDataset, visualize_samples
import random
from util.plotter import Plotter, _step_png_path
from datetime import datetime
import os
import threading
import json
import torch.nn.functional as F
import torch
from collections import defaultdict





class BucketBatchSampler:
    """
    Groups dataset samples into batches by scene width so that every batch contains
    same-width scenes. This allows training on datasets with variable-width scenes
    since torch.stack requires uniform shapes within a batch.

    Args:
        dataset: A LevelDataset whose samples are (scene_tensor, ...) with scene shape (Channels, H, W)
        batch_size (int): Number of samples per batch
        drop_last (bool): If True, discard incomplete batches at the end of each width bucket
        shuffle (bool): If True, shuffle samples within buckets and shuffle the batch order

    Attributes:
        shapes (list[int]): Each unique level width present in the dataset, used for generating
            samples of different size at epoch benchmarks during training.
    """
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True):
        self.shuffle = shuffle
        # Group dataset indices by scene width
        buckets = defaultdict(list)
        for idx in range(len(dataset)):
            w = dataset[idx][0].shape[2]  # scene tensor is (C, H, W)
            buckets[w].append(idx)

        self.batches = []
        for indices in buckets.values():
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                # Skip incomplete batches at the tail of each bucket when drop_last is set
                if drop_last and len(batch) < batch_size:
                    continue
                self.batches.append(batch)

        # Unique scene widths present in the dataset; used to generate variably-sized benchmark samples at epoch checkpoints during training
        self.shapes = list(buckets.keys())

    def __iter__(self):
        # Re-shuffle every epoch so sample order varies across epochs, matching DataLoader(shuffle=True) behavior.
        # A uniform-width dataset produces one bucket, making this equivalent to the standard DataLoader shuffle path.
        if self.shuffle:
            batches = self.batches.copy()
            random.shuffle(batches)
            return iter(batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
    
    


def create_dataloaders(json_path, val_json, tokenizer, data_mode, augment, num_tiles,
                       negative_prompt_training, block_embeddings, batch_size,
                       persistent_workers=True, multiple_captions=False):
    """
    Create PyTorch dataloaders for training and validation datasets.

    Args:
        json_path (str): Path to the training dataset JSON file.
        val_json (str or None): Path to the validation dataset JSON file, or None to skip validation.
        tokenizer: Tokenizer to use for processing captions.
        mode (str): "text" for just the text captions, 
                    "diff_text" for level scenes and text captions (used with a pretrained model).
        augment (bool): Whether to apply data augmentation to the training dataset.
        num_tiles (int): Number of tiles to use in the level representation (for "diff_text" mode).
        negative_prompt_training (bool): Whether to include negative captions for training.
        block_embeddings (torch.Tensor or None): Precomputed block embeddings for "diff_text" mode, or None if not using.
        batch_size (int): Batch size for the dataloaders.
        multiple_captions (bool): If True, the training set selects one of each sample's stored
            captions ("caption", "caption1", ...) at random per access, in place of phrase-shuffle
            augmentation. Validation always uses the canonical "caption" deterministically.

    Returns:
        tuple(train_dataloader, val_dataloader, sample_widths): where sample_widths is the
            list of unique scene widths in the training set, used to generate variably-sized
            benchmark samples at epoch checkpoints.
    """

    # Initialize dataset
    train_dataset = LevelDataset(
        json_path=json_path,
        tokenizer=tokenizer,
        shuffle=True,
        mode=data_mode,
        augment=augment,
        num_tiles=num_tiles,
        negative_captions=negative_prompt_training,
        block_embeddings=block_embeddings,
        multiple_captions=multiple_captions
    )
    val_dataset = None
    if val_json is not None:
        val_dataset = LevelDataset(
            json_path=val_json,
            tokenizer=tokenizer,
            shuffle=False,
            mode=data_mode,
            augment=False,
            num_tiles=num_tiles,
            negative_captions=negative_prompt_training,
            block_embeddings=block_embeddings
        )

    # BucketBatchSampler groups same-width scenes into each batch, allowing mixed-width datasets.
    # batch_size/shuffle/drop_last are owned by the sampler, not passed directly to DataLoader.
    train_sampler = BucketBatchSampler(train_dataset, batch_size, drop_last=True, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        persistent_workers=persistent_workers
    )

    val_dataloader = None
    if val_dataset is not None:
        # drop_last=False so all validation samples are evaluated regardless of bucket remainder size
        val_dataloader = DataLoader(
            val_dataset,
            batch_sampler=BucketBatchSampler(val_dataset, batch_size, drop_last=False, shuffle=False),
            num_workers=4,
            persistent_workers=True
        )

    # Unique training-set scene widths, used to benchmark variably-sized samples during training.
    return train_dataloader, val_dataloader, train_sampler.shapes


def get_random_training_samples(train_dataloader, negative_prompt_training, output_dir = None, game = 'Mario', block_embeddings = None):
    """
    Get random training samples from the dataloader and print them to the console.
    Args:
        train_dataloader: The PyTorch dataloader for the training dataset.
        negative_prompt_training (bool): Whether the dataset includes negative captions.
        output_dir (str or None): If provided, a directory to save the sample captions to a text file.
            The source scene images are also saved to a "samples_original" subdirectory so the
            generated samples produced during training can be compared against them.
        game (str): Game whose tile set is used to render the source scene images.
        block_embeddings (Tensor or None): Block embeddings used to decode the scene tensors when
            the dataset is embedding-based (otherwise scenes are one-hot encoded).

    Returns:
        sample_captions (list of str): A list of randomly sampled captions from the training dataset.
        sample_negative_captions (list of str or str): If negative_prompt_training is True, a list of randomly sampled negative captions. Otherwise, an empty string.
    """

    train_dataset = train_dataloader.dataset
    # Sample four random captions from the dataset
    sample_indices = [random.randint(0, len(train_dataset) - 1) for _ in range(4)]

    # Fetch each sample once so the caption, negative caption, and saved scene image all
    # correspond to the same item (__getitem__ re-augments/flips on every access).
    sample_items = [train_dataset[i] for i in sample_indices]

    sample_captions = [item[1] for item in sample_items]
    print("Sample captions:")
    for caption in sample_captions:
        print(caption)

    sample_negative_captions = ""
    if negative_prompt_training:
        sample_negative_captions = [item[2] for item in sample_items]
        print("Sample negative captions:")
        for caption in sample_negative_captions:
            print(f"  NEG: {caption}")

    #Write captions to a file
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "sample_captions.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("Sample captions:\n")
            for caption in sample_captions:
                f.write(str(caption) + "\n")
            if negative_prompt_training:
                f.write("\nSample negative captions:\n")
                for caption in sample_negative_captions:
                    f.write(str(caption) + "\n")
        print(f"Sample captions written to {out_path}")

        # Save the source scene image for each sampled caption so generated samples can be
        # compared against their ground-truth scenes. Scenes are rendered one at a time (rather
        # than as a batch) so datasets with variable scene widths are handled. The filenames
        # mirror visualize_samples' generated-sample naming (sample_{i} - {prompt}.png).
        originals_dir = os.path.join(output_dir, "samples_original")
        for i, item in enumerate(sample_items):
            scene = item[0].unsqueeze(0)  # add batch dim: [1, channels, height, width]
            visualize_samples(
                scene,
                originals_dir,
                start_index=i,
                prompts=[sample_captions[i]],
                game=game,
                block_embeddings=block_embeddings,
            )
        print(f"Sample source scenes written to {originals_dir}")


    return sample_captions, sample_negative_captions


def start_plotter(log_file, output_dir, left_key, right_key, left_label, right_label, png_name):
    formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')

    epoch_png = f'{png_name}_{formatted_date}.png'
    plotter = Plotter(log_file, update_interval=5.0, left_key=left_key, right_key=right_key,
                            left_label=left_label, right_label=right_label, output_png=epoch_png)
    plot_thread = threading.Thread(target=plotter.start_plotting)
    plot_thread.daemon = True
    plot_thread.start()
    print(f"{png_name} plotting enabled.")
    print(f"  Epoch-based plot : {os.path.join(output_dir, epoch_png)}")
    print(f"  Step-based plot  : {os.path.join(output_dir, _step_png_path(epoch_png))}")
    return plotter, plot_thread


def kill_plotter(plotter, plot_thread):
    if plot_thread and plot_thread.is_alive():
        plotter.stop_plotting()
        plot_thread.join(timeout=5.0)
        if plot_thread.is_alive():
            print("Warning: Plot thread did not terminate properly")


def load_config_from_json(config_path):
    """Load hyperparameters from a JSON config file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Configuration loaded from {config_path}")
            
            # Print the loaded config for verification
            print("Loaded hyperparameters:")
            for key, value in config.items():
                print(f"  {key}: {value}")
                
            return config
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading config file: {e}")
        raise e


def update_args_from_config(args, config):
    """Update argparse namespace with values from config."""
    # Convert config dict to argparse namespace
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def get_scene_from_embeddings(image, block_embeddings):
    """Code copied over from level_dataset, should give limited support for block embeddings"""
    # Reshape sample to [batch_size * height * width, embedding_dim]
    batch_size, embedding_dim, height, width = image.shape
    
    flat_samples = image.permute(0, 2, 3, 1).reshape(-1, embedding_dim)
    
    # Normalize vectors for cosine similarity
    flat_samples = F.normalize(flat_samples, p=2, dim=1).cpu()
    block_embeddings = F.normalize(block_embeddings, p=2, dim=1)

    # Calculate cosine similarity between each position and all tile embeddings
    similarities = torch.matmul(flat_samples, block_embeddings.t())
    
    # Get indices of most similar tiles
    indices = torch.softmax(similarities, dim=1)


    # Reshape back to [batch_size, height, width, num_tiles]
    num_tiles = block_embeddings.shape[0]
    indices = indices.reshape(batch_size, height, width, num_tiles)
    indices = indices.permute(0, 3, 1, 2)

    image=indices.detach().cpu()
    return image