import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from level_dataset import LevelDataset
from tokenizer import Tokenizer
from models.text_model import TransformerModel
from evaluate_masked_token_prediction import evaluate_model, masked_inputs
import json
import os
import threading
from datetime import datetime
from util.loss_plotter import LossPlotter
import random

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

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, tokenizer, patience=20):
    global args

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
    plotter = LossPlotter(log_file, update_interval=5.0, left_key='loss', right_key='val_loss', left_label='Loss', right_label='Val Loss')
    plot_thread = threading.Thread(target=plotter.start_plotting)
    plot_thread.daemon = True
    plot_thread.start()

    # Add function to log metrics
    def log_metrics(epoch, loss, lr, val_loss=None, step=None):
        log_entry = {
            "epoch": epoch,
            "loss": loss,
            "val_loss": val_loss,
            "lr": lr,
            "step": step if step is not None else epoch * len(train_loader),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')    

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    best_model_state = None  # Add this

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Masking: Replace some tokens with [MASK] (handled in dataset or here)
            input_batch, target_batch = batch.clone(), batch.clone()
            input_batch = masked_inputs(input_batch, tokenizer, device=device)
            
            output = model(input_batch)

            loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation loss
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_total = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device)
                    input_batch, target_batch = val_batch.clone(), val_batch.clone()
                    input_batch = masked_inputs(input_batch, tokenizer, device=device)
                    output = model(input_batch)
                    loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
                    val_loss_total += loss.item()
            val_loss = val_loss_total / len(val_loader)
            model.train()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model state
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered. Best validation loss: {best_val_loss:.4f}")
                    # Restore best model state
                    model.load_state_dict(best_model_state['model_state_dict'])
                    early_stop = True
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Log to JSONL file
        log_metrics(epoch, avg_loss, args.lr, val_loss=val_loss)

        # Save checkpoint if enabled and at the correct interval
        if args.save_checkpoints and args.checkpoint_freq > 0 and (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")

            if early_stop:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
                break

    plotter.stop_plotting()
    evaluate_model(model, tokenizer, train_loader, device)
    
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
    parser.add_argument("--checkpoint_freq", type=int, default=20, help="Save checkpoint every N epochs (0 to disable)")
    parser.add_argument("--save_checkpoints", action="store_true", help="Enable periodic checkpoint saving")
    parser.add_argument('--split', action='store_true', help='Enable train/val/test split')
    parser.add_argument('--train_pct', type=float, default=0.7, help='Train split percentage (default 0.7)')
    parser.add_argument('--val_pct', type=float, default=0.1, help='Validation split percentage (default 0.1)')
    parser.add_argument('--test_pct', type=float, default=0.2, help='Test split percentage (default 0.2)')
    parser.add_argument("--patience", type=int, default=20, help="Number of epochs to wait for improvement in val loss before early stopping (default: 20)")
    
    global args
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    tokenizer.load(args.pkl)
    
    if args.split:
        # Ensure percentages sum to 1.0
        total = args.train_pct + args.val_pct + args.test_pct
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split percentages must sum to 1.0, got {total}")
        train_json, val_json, test_json = split_dataset(args.json, args.train_pct, args.val_pct, args.test_pct)
        train_dataset = LevelDataset(train_json, tokenizer, mode="mlm", augment=args.augment, limit=args.data_limit)
        val_dataset = LevelDataset(val_json, tokenizer, mode="mlm", augment=False, limit=-1)
        test_dataset = LevelDataset(test_json, tokenizer, mode="mlm", augment=False, limit=-1)
    else:
        train_dataset = LevelDataset(args.json, tokenizer, mode="mlm", augment=args.augment, limit=args.data_limit)
        val_dataset = None
        test_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) if test_dataset is not None else None
    
    print(f"Num train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Num val samples: {len(val_dataset)}")
    if test_dataset is not None:
        print(f"Num test samples: {len(test_dataset)}")
    print(f"Num train batches: {len(train_loader)}")
    if val_loader is not None:
        print(f"Num val batches: {len(val_loader)}")
    if test_loader is not None:
        print(f"Num test batches: {len(test_loader)}")

    vocab_size = tokenizer.get_vocab_size()
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    
    model = TransformerModel(vocab_size, embedding_dim, hidden_dim, tokenizer).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["[PAD]"])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, tokenizer, patience=args.patience)
    model.save_pretrained(args.output_dir)
    print(f"Model saved in {args.output_dir}")

    # Final evaluation on all splits
    print("\nFinal evaluation:")
    if args.split:
        print("Train set:")
        evaluate_model(model, tokenizer, train_loader, device)
        print("Validation set:")
        evaluate_model(model, tokenizer, val_loader, device)
        print("Test set:")
        evaluate_model(model, tokenizer, test_loader, device)
    else:
        print("Full data:")
        evaluate_model(model, tokenizer, train_loader, device)
