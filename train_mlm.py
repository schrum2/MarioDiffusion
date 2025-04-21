import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from level_dataset import LevelDataset
from tokenizer import Tokenizer
from models import TransformerModel
from masked_token_prediction import evaluate_model, masked_inputs
import json
import os
import threading
import time
from datetime import datetime
from util.loss_plotter import LossPlotter

def train(model, dataloader, criterion, optimizer, device, epochs, tokenizer):

    #for batch in dataloader:
    #    print("Batch shape:", batch.shape)
    #    # Decode and print first few sequences in the batch
    #    for seq in batch:
    #        decoded_seq = tokenizer.decode(seq.tolist())
    #        print(decoded_seq)
    #quit()

    # Get formatted timestamp for filenames
    formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # Create log files
    log_file = os.path.join(args.output_dir, f"mlm_training_log_{formatted_date}.jsonl")
    config_file = os.path.join(args.output_dir, f"hyperparams_{formatted_date}.json")

    # Save hyperparameters to JSON file
    hyperparams = vars(args)
    with open(config_file, "w") as f:
        json.dump(hyperparams, f, indent=4)
    print(f"Saved configuration to: {config_file}")

    plotter = None
    plot_thread = None
    plotter = LossPlotter(log_file, update_interval=5.0)  # Update every 5 seconds
    plot_thread = threading.Thread(target=plotter.start_plotting)
    plot_thread.daemon = True
    plot_thread.start()

    # Add function to log metrics
    def log_metrics(epoch, loss, lr, step=None):
        log_entry = {
            "epoch": epoch,
            "loss": loss,
            "lr": lr,
            "step": step if step is not None else epoch * len(dataloader),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')    

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Masking: Replace some tokens with [MASK] (handled in dataset or here)
            input_batch, target_batch = batch.clone(), batch.clone()
            input_batch = masked_inputs(input_batch, tokenizer, device=device)
            
            output = model(input_batch)

            #print(f"Output shape: {output.shape}")
            #print(f"Target batch shape: {target_batch.shape}")

            loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Log to JSONL file
        log_metrics(epoch, avg_loss, args.lr)

    evaluate_model(model, tokenizer, dataloader, device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Length of text embedding vectors")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Units in hidden layers")
    parser.add_argument("--batch_size", type=int, default=16, help="Training samples per batch")
    parser.add_argument("--data_limit", type=int, default=-1, help="If not negative, only train with this many examples")
    parser.add_argument("--output_dir", type=str, default="mlm", help="Directory for training logs and model")
    parser.add_argument('--no-augment', action='store_false', dest='augment', help='Disable data augmentation (default: True)')
    
    global args
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    tokenizer.load(args.pkl)
    dataset = LevelDataset(args.json, tokenizer, mode="mlm", augment=args.augment, limit=args.data_limit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"Num samples: {len(dataset)}")
    print(f"Num batches: {len(dataloader)}")

    vocab_size = tokenizer.get_vocab_size()
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    
    model = TransformerModel(vocab_size, embedding_dim, hidden_dim, tokenizer).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["[PAD]"])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    train(model, dataloader, criterion, optimizer, device, args.epochs, tokenizer)
    model.save_pretrained(args.output_dir)
    print(f"Model saved in {args.output_dir}")
