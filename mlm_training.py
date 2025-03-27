import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from level_dataset import LevelDataset
from tokenizer import Tokenizer
from models import LSTMModel, TransformerModel

def train(model, dataloader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Masking: Replace some tokens with [MASK] (handled in dataset or here)
            input_batch, target_batch = batch.clone(), batch.clone()
            mask_token = tokenizer.token_to_id["[MASK]"]
            mask_prob = 0.15  # Standard MLM masking probability
            mask = torch.rand(input_batch.shape, device=device) < mask_prob
            input_batch[mask] = mask_token
            
            output = model(input_batch)
            loss = criterion(output.view(-1, output.size(-1)), target_batch.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer"], required=True, help="Model type")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Length of text embedding vectors")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    tokenizer.load(args.pkl)
    dataset = LevelDataset(args.json, tokenizer, batch_size=16, mode="mlm")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    vocab_size = tokenizer.get_vocab_size()
    embedding_dim = args.embedding_dim
    hidden_dim = 256  # Adjustable
    
    if args.model == "lstm":
        model = LSTMModel(vocab_size, embedding_dim, hidden_dim).to(device)
        model_name = "mlm_lstm.pth"
    else:
        model = TransformerModel(vocab_size, embedding_dim, hidden_dim).to(device)
        model_name = "mlm_transformer.pth"
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["[PAD]"])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train(model, dataloader, criterion, optimizer, device, args.epochs)
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")
