import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup 
from tqdm.auto import tqdm
import random
import numpy as np
from accelerate import Accelerator
from level_dataset import LevelDataset, visualize_samples
from tokenizer import Tokenizer 
import json
import threading
from datetime import datetime
from util.plotter import Plotter
from models.text_model import TransformerModel
from evaluate_caption_adherence import calculate_caption_score_and_samples
from create_ascii_captions import extract_tileset
from transformers import AutoTokenizer, AutoModel
from models.fdm import Gen


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text-conditional diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--pkl", type=str, default=None, help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_tiles", type=int, default=15, help="Number of tile types")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument('--split', action='store_true', help='Enable train/val/test split') # TODO: Allow SMB1 data to be split into groups for training and testing
    parser.add_argument('--train_pct', type=float, default=0.9, help='Train split percentage (default 0.9)')
    parser.add_argument('--val_pct', type=float, default=0.05, help='Validation split percentage (default 0.05)')
    parser.add_argument('--test_pct', type=float, default=0.05, help='Test split percentage (default 0.05)')
    
    # New text conditioning args
    parser.add_argument("--mlm_model_dir", type=str, default="mlm", help="Path to pre-trained text embedding model")
    parser.add_argument("--pretrained_language_model", type=str, default=None, help="Link to a pre-trained language model, everything after huggingface.co/. This will override the mlm_model_dir argument.")
    
    # Model args
    parser.add_argument("--embedding_dim", type=int, default=384, help="Base size of the embedded tokens into the model")#TODO: not sure if this can be changed
    parser.add_argument("--z_dim", type=int, default=5, help="Size of the concatenated noise vector for varitey (Default 5)")
    parser.add_argument("--kern_size", type=int, default=7, help="Kernel size for convolutional layers (default 7)")
    parser.add_argument("--filter_count", type=int, default=128, help="Number of filters in the convolutional layers (default 128)")
    parser.add_argument("--num_res_blocks", type=int, default=3, help="Number of residual blocks in the generator (default 3)")


    # Training args
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size") # TODO: Consider reducing to 16 to help generalization
    parser.add_argument("--save_image_epochs", type=int, default=20, help="Save generated levels every N epochs")
    parser.add_argument("--save_model_epochs", type=int, default=20, help="Save model every N epochs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate_epochs", type=int, default=5, help="Calculate validation loss every N epochs")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="level-fdm-output", help="Output directory")

    # For caption score calculation
    parser.add_argument("--tileset", default='..\TheVGLC\Super Mario Bros\smb.json', help="Descriptions of individual tile types")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Indicate when there are no occurrences of an item or structure")
    parser.add_argument("--plot_validation_caption_score", action="store_true", default=False, help="Whether validation caption score should be plotted")

    # For block2vec embedding model
    parser.add_argument("--block_embedding_model_path", type=str, default=None, help="Path to trained block embedding model (.pt)")



    return parser.parse_args()



class imageDataSet(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self,idx):
        return self.data[0][idx], self.data[1][idx]




def main():
    args = parse_args()

    # Check if output directory already exists
    if os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' already exists. Please remove it or choose a different name.")
        exit()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Setup accelerator
    accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
        )

    # Initialize tokenizer
    if args.pkl:
        tokenizer = Tokenizer()
        tokenizer.load(args.pkl)
    else:
        tokenizer = None

    # Load text embedding model if text conditioning is enabled
    if args.pretrained_language_model: #Default to huggingface model, if it exists
        text_encoder = AutoModel.from_pretrained(args.pretrained_language_model, trust_remote_code=True).to(accelerator.device)
        text_encoder.eval() # Set to evaluation mode
        model_embedding_dim = text_encoder.config.hidden_size#TODO: not sure if this can be changed
        tokenizer_hf = AutoTokenizer.from_pretrained(args.pretrained_language_model)
        print(f"Loaded text encoder from {args.pretrained_language_model}")
    elif args.mlm_model_dir:
        text_encoder = TransformerModel.from_pretrained(args.mlm_model_dir).to(accelerator.device)
        text_encoder.eval()  # Set to evaluation mode
        model_embedding_dim = text_encoder.embedding_dim#TODO: not sure if this can be changed
        tokenizer_hf = None #We don't need the huggingface tokenizer if we're using our own, varible initialization done to avoid future errors
        print(f"Loaded text encoder from {args.mlm_model_dir}")
    
    data_mode = "diffusion" if not args.pretrained_language_model else "diff_text"

    # Load block embedding model if specified
    block_embeddings = None
    embedding_dim = None
    if args.block_embedding_model_path:
        try:
            # Load embeddings from the embeddings.pt file in the model directory
            block_embeddings = torch.load(
                os.path.join(args.block_embedding_model_path, "embeddings.pt"),
                map_location=accelerator.device
            )
            
            embedding_dim = block_embeddings.shape[1]
            print(f"Loaded block embeddings from {args.block_embedding_model_path} with dimension {embedding_dim}")
        except Exception as e:
            print(f"Error loading block embedding model: {e}")
            raise
    

     # Initialize dataset
    if args.split:
        train_json, val_json, test_json = split_dataset(args.json, args.train_pct, args.val_pct, args.test_pct)
        train_dataset = LevelDataset(
            json_path=train_json,
            tokenizer=tokenizer,
            shuffle=True,
            mode=data_mode,
            augment=args.augment,
            num_tiles=args.num_tiles,
            block_embeddings=block_embeddings
        )
        val_dataset = LevelDataset(
            json_path=val_json,
            tokenizer=tokenizer,
            shuffle=False,
            mode=data_mode,
            augment=False,
            num_tiles=args.num_tiles,
            block_embeddings=block_embeddings
        )
    else:
        train_dataset = LevelDataset(
            json_path=args.json,
            tokenizer=tokenizer,
            shuffle=True,
            mode=data_mode,
            augment=args.augment,
            num_tiles=args.num_tiles,
            block_embeddings=block_embeddings
        )
        val_dataset = None

    first_sample = train_dataset[0]
    scene_height = first_sample[0].shape[1]
    scene_width = first_sample[0].shape[2]

    print(f"Scene height: {scene_height}")
    print(f"Scene width: {scene_width}")

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )


        # Sample four random captions from the dataset
        sample_indices = [random.randint(0, len(train_dataset) - 1) for _ in range(4)]
        # Original code for positive-only captions
        sample_embedding_vectors = [train_dataset[i][1] for i in sample_indices]
        if not args.pretrained_language_model:
            sample_embedding_vectors = [v.tolist() for v in sample_embedding_vectors]
            pad_token = tokenizer.token_to_id["[PAD]"]
            sample_captions = [
                tokenizer.decode([token for token in caption if token != pad_token]) 
                for caption in sample_embedding_vectors
            ]
        else:
            sample_captions = [
                v for v in sample_embedding_vectors
            ]
        print("Sample captions:")
        for caption in sample_captions:
            print(caption)
    

    #Create an instance of the model
    model = Gen(
        model_name=args.mlm_model_dir,
        embedding_dim=args.embedding_dim,
        z_dim=args.z_dim,
        kern_size=args.kern_size,
        filter_count=args.filter_count,
        num_res_blocks=args.num_res_blocks
    )
    print(f"Model: {model}")

    # if there is no block embedding model, set the channels to num_tiles
    in_channels = embedding_dim if args.block_embedding_model_path else args.num_tiles
    # else set channels to the embedding dimension of the model
    out_channels = in_channels

    print(train_dataloader)
    print(train_dataloader.dataset)
    print(train_dataloader.dataset[0][0].shape)
    print(train_dataloader.dataset[0][1].shape)
    for embeddings, ytrue in train_dataloader:
        print(embeddings.shape)
        print(ytrue.shape)

    train(model, args.num_epochs, train_dataloader, val_dataloader, None, accelerator.device)
    #Required steps:
    #Setup model, new Gen object
    #Setup dataset
        #Split dataset into train/val/test
            #Generally follow fdm.py
    #Setup tokenizer
        #Get encoding of a word (length of embedding_dim)
        #Get noise vector (length of z_dim)
        #Concat
    #Setup training loop
        #follow the train method



def train(model, EPOCHS, train_set, val_set, test_set, device):

    
    loss_metric_train = torch.zeros(EPOCHS).to(device)
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(EPOCHS):
        
        for embeddings, ytrue in train_set:
            optimizer.zero_grad()
            outputs = model(embeddings.to(device), torch.rand(len(embeddings), 5).to(device))

            loss = torch.nn.NLLLoss()(torch.log(outputs), ytrue.argmax(dim=3).to(device))
            #loss = torch.nn.functional.mse_loss(torch.log(outputs), ytrue.argmax(dim=1).to(device))

            loss_metric_train[epoch] += loss

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss_metric_train[epoch].item()}")


def split_dataset(json_path, train_pct, val_pct, test_pct):
    """Splits the dataset into train/val/test and saves them as new JSON files."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    n = len(data)
    indices = list(range(n))
    random.shuffle(indices)
    train_end = int(train_pct * n)
    val_end = train_end + int(val_pct * n)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
    base, ext = os.path.splitext(json_path)
    train_path = f"{base}-train{ext}"
    val_path = f"{base}-validate{ext}"
    test_path = f"{base}-test{ext}"
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    return train_path, val_path, test_path


if __name__ == "__main__":
    main()