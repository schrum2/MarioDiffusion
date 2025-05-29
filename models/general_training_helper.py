from torch.utils.data import DataLoader
from level_dataset import LevelDataset
import random
from util.plotter import Plotter
from datetime import datetime
import os
import threading
import json



def create_dataloaders(json_path, val_json, tokenizer, data_mode, augment, num_tiles, 
                       negative_prompt_training, block_embeddings, batch_size):
    # Initialize dataset
    train_dataset = LevelDataset(
        json_path=json_path,
        tokenizer=tokenizer,
        shuffle=True,
        mode=data_mode,
        augment=augment,
        num_tiles=num_tiles,
        negative_captions=negative_prompt_training,
        block_embeddings=block_embeddings
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

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
    
    return train_dataloader, val_dataloader


def get_random_training_samples(train_dataloader, negative_prompt_training, output_dir = None):
    train_dataset = train_dataloader.dataset
    # Sample four random captions from the dataset
    sample_indices = [random.randint(0, len(train_dataset) - 1) for _ in range(4)]

    sample_captions = [train_dataset[i][1] for i in sample_indices]
    print("Sample captions:")
    for caption in sample_captions:
        print(caption)

    sample_negative_captions = ""
    if negative_prompt_training:
        sample_negative_captions = [train_dataset[i][2] for i in sample_indices]
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


    return sample_captions, sample_negative_captions


def start_plotter(log_file, output_dir, left_key, right_key, left_label, right_label, png_name):
    formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')

    plotter = Plotter(log_file, update_interval=5.0, left_key=left_key, right_key=right_key,
                            left_label=left_label, right_label=right_label, output_png=f'{png_name}_{formatted_date}.png')
    plot_thread = threading.Thread(target=plotter.start_plotting)
    plot_thread.daemon = True
    plot_thread.start()
    print(f"Loss plotting enabled. Progress will be saved to {os.path.join(output_dir, f'{png_name}_{formatted_date}.png')}")
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


