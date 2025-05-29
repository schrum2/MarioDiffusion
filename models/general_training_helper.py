from torch.utils.data import DataLoader
from level_dataset import LevelDataset
import random


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


def get_random_training_samples(train_dataloader, negative_prompt_training):
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
    return sample_captions, sample_negative_captions


