import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from level_dataset import LevelDataset
from tokenizer import Tokenizer
from models import LSTMModel, TransformerModel
from masked_token_prediction import evaluate_model, masked_inputs

def train(model, dataloader, criterion, optimizer, device, epochs, tokenizer):

    #for batch in dataloader:
    #    print("Batch shape:", batch.shape)
    #    # Decode and print first few sequences in the batch
    #    for seq in batch:
    #        decoded_seq = tokenizer.decode(seq.tolist())
    #        print(decoded_seq)
    #quit()

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
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    evaluate_model(model, tokenizer, dataloader, device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer"], required=True, help="Model type")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Length of text embedding vectors")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Units in hidden layers")
    parser.add_argument("--data_limit", type=int, default=-1, help="If not negative, only train with this many examples")
    parser.add_argument('--no-augment', action='store_false', dest='augment', help='Disable data augmentation (default: True)')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    tokenizer.load(args.pkl)
    dataset = LevelDataset(args.json, tokenizer, batch_size=16, mode="mlm", augment=args.augment, limit=args.data_limit)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"Num batches: {len(dataset)}")

    vocab_size = tokenizer.get_vocab_size()
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    
    if args.model == "lstm":
        model = LSTMModel(vocab_size, embedding_dim, hidden_dim).to(device)
        model_name = "mlm_lstm.pth"
    else:
        model = TransformerModel(vocab_size, embedding_dim, hidden_dim).to(device)
        model_name = "mlm_transformer.pth"
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["[PAD]"])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    train(model, dataloader, criterion, optimizer, device, args.epochs, tokenizer)
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")
