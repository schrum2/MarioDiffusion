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
from util.plotter import Plotter
import random
import models.text_model as text_model

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, tokenizer, patience=20):
    global args

    # Get formatted timestamp for filenames
    formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create log files
    log_file = os.path.join(args.output_dir, f"mlm_training_log_{formatted_date}.jsonl")
    accuracy_log_file = os.path.join(args.output_dir, f"mlm_accuracy_log_{formatted_date}.jsonl")
    config_file = os.path.join(args.output_dir, f"hyperparams_{formatted_date}.json")

    # Save hyperparameters to JSON file
    hyperparams = vars(args)
    with open(config_file, "w") as f:
        json.dump(hyperparams, f, indent=4)
    print(f"Saved configuration to: {config_file}")

    # Create two plotters - one for loss, one for accuracy
    loss_plotter = Plotter(log_file, update_interval=5.0, left_key='loss', right_key='val_loss', 
                           left_label='Loss', right_label='Val Loss', 
                           output_png=f'training_loss_{formatted_date}.png')
    accuracy_plotter = Plotter(accuracy_log_file, update_interval=5.0, left_key='train_accuracy', 
                               right_key='val_accuracy', left_label='Train Accuracy', right_label='Val Accuracy',
                               output_png=f'training_accuracy_{formatted_date}.png')
    
    # Start plotting threads
    loss_plot_thread = threading.Thread(target=loss_plotter.start_plotting)
    loss_plot_thread.daemon = True
    loss_plot_thread.start()
    
    accuracy_plot_thread = threading.Thread(target=accuracy_plotter.start_plotting)
    accuracy_plot_thread.daemon = True
    accuracy_plot_thread.start()

    # Add functions to log metrics
    def log_loss_metrics(epoch, loss, lr, val_loss=None, step=None):
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

    def log_accuracy_metrics(epoch, train_accuracy, val_accuracy=None, step=None):
        log_entry = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "step": step if step is not None else epoch * len(train_loader),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(accuracy_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    best_model_state = None
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, 
        patience=patience//2, min_lr=1e-6
    )

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in progress_bar:
            batch = text_model.encode_token_captions(batch, tokenizer, model.max_seq_length, device=device)
            optimizer.zero_grad()

            
            # Masking: Replace some tokens with [MASK] (handled in dataset or here)
            input_batch, target_batch = batch.clone(), batch.clone()
            input_batch = masked_inputs(input_batch, tokenizer, device=device)
            
            output = model(input_batch)

            loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                'loss': loss.item(),
                **(({'no_improve': f'{epochs_no_improve}/{patience}'} if args.use_early_stopping else {}))
            })
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation loss
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_total = 0
            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f"Validation", leave=False)
                for val_batch in val_progress:
                    val_batch = text_model.encode_token_captions(val_batch, tokenizer, model.max_seq_length, device=device)
                    input_batch, target_batch = val_batch.clone(), val_batch.clone()
                    input_batch = masked_inputs(input_batch, tokenizer, device=device)
                    output = model(input_batch)
                    loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
                    val_loss_total += loss.item()
                    val_progress.set_postfix(loss=loss.item())
            val_loss = val_loss_total / len(val_loader)
            model.train()

            
            status_msg = f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}"
            if args.use_early_stopping:
                status_msg += f", No Improvement: {epochs_no_improve}/{patience}"
            print(status_msg)
            
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
            elif args.use_early_stopping:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered. Best validation loss: {best_val_loss:.4f}")
                    early_stop = True
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Log to JSONL file
        log_loss_metrics(epoch, avg_loss, args.lr, val_loss=val_loss)

        # Update learning rate scheduler
        if val_loader is not None:
            scheduler.step(val_loss)

        # Save checkpoint if enabled and at the correct interval
        if args.save_checkpoints and args.checkpoint_freq > 0 and (epoch + 1) % args.checkpoint_freq == 0:
            # Evaluate model on train and validation sets
            train_accuracy, train_correct, train_total = evaluate_model(model, tokenizer, train_loader, device, console_output=False)
            val_accuracy = float("nan")
            if val_loader is not None:
                val_accuracy, val_correct, val_total = evaluate_model(model, tokenizer, val_loader, device, console_output=False)
            
            # Log accuracies
            log_accuracy_metrics(epoch, train_accuracy, val_accuracy)
            
            # Save checkpoint
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pt'))
            model.save_pretrained(checkpoint_dir)  # Save model config separately
            print(f"Saved checkpoint to {checkpoint_dir} (Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%)")
        
        if args.use_early_stopping and early_stop:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break

    loss_plotter.stop_plotting()
    accuracy_plotter.stop_plotting()

    # Restore best model state
    # At end of training, always restore best model if one was saved
    if best_model_state:
        print(f"\nTraining complete. Restoring best model from epoch {best_model_state['epoch']} with validation loss {best_val_loss:.4f}")
        model.load_state_dict(best_model_state['model_state_dict'])
    
    evaluate_model(model, tokenizer, train_loader, device, console_output = False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--val_json", type=str, default=None, help="Optional path to validation dataset json file (determines early stopping)")
    parser.add_argument("--test_json", type=str, default=None, help="Optional path to testing dataset json file (used at end of training)")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Length of text embedding vectors")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Units in hidden layers")
    parser.add_argument("--batch_size", type=int, default=16, help="Training samples per batch")
    parser.add_argument("--data_limit", type=int, default=-1, help="If not negative, only train with this many examples")
    parser.add_argument("--output_dir", type=str, default="mlm", help="Directory for training logs and model")
    parser.add_argument('--no_augment', action='store_false', dest='augment', help='Disable data augmentation (default: True)')
    parser.add_argument("--checkpoint_freq", type=int, default=20, help="Save checkpoint every N epochs (0 to disable)")
    parser.add_argument("--save_checkpoints", action="store_true", help="Enable periodic checkpoint saving")
    parser.add_argument("--patience", type=int, default=30, help="Number of epochs to wait for improvement in val loss before early stopping (default: 20)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--use_early_stopping", action="store_true", help="Stop training if validation/caption performance stagnate")
    
    global args
    args = parser.parse_args()

    # Check if the output directory exists
    if os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' already exists. Please remove it or choose a different name.")
        exit()
    
    # Set random seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    tokenizer.load(args.pkl)
    
    train_dataset = LevelDataset(args.json, tokenizer, mode="text", augment=args.augment, limit=args.data_limit)
    val_dataset = None
    if args.val_json:
        val_dataset = LevelDataset(args.val_json, tokenizer, mode="text", augment=False, limit=-1, shuffle=False, )
    test_dataset = None
    if args.test_json:
        test_dataset = LevelDataset(args.test_json, tokenizer, mode="text", augment=False, limit=-1, shuffle=False, )

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
    print("Train set:")
    evaluate_model(model, tokenizer, train_loader, device)
    if val_loader:
        print("Validation set:")
        evaluate_model(model, tokenizer, val_loader, device)
    if test_loader:
        print("Test set:")
        evaluate_model(model, tokenizer, test_loader, device)
