import argparse
import torch
import random
from tokenizer import Tokenizer
from models import LSTMModel, TransformerModel
from level_dataset import LevelDataset
from torch.utils.data import DataLoader

def masked_inputs(input_batch, tokenizer, device, mask_prob=0.15):
    mask_token = tokenizer.token_to_id["[MASK]"]
    pad_token = tokenizer.token_to_id["[PAD]"] # Don't mask [PAD] tokens
    # Create a deep copy to avoid modifying the original batch
    masked_batch = input_batch.clone().to(device)
    # Create a mask that respects [PAD] tokens
    mask = (torch.rand(masked_batch.shape, device=device) < mask_prob) & (masked_batch != pad_token)
    # Replace tokens with mask token
    masked_batch[mask] = mask_token
    
    return masked_batch, mask  # Return both masked input and mask for loss calculation

def evaluate_model(model, tokenizer, dataloader, device, mask_prob=0.15, verbose=True):
    model.eval()
    mask_token = tokenizer.token_to_id["[MASK]"]
    pad_token = tokenizer.token_to_id["[PAD]"]
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            masked_batch, mask = masked_inputs(batch, tokenizer, device, mask_prob)
            
            # Perform forward pass
            output = model(masked_batch)
            
            # Reshape output and target for comparison
            # output shape: [batch_size, seq_len, vocab_size]
            # batch shape: [batch_size, seq_len]
            
            # Get predictions
            predictions = output.argmax(dim=-1)
            
            # Mask out padding and non-masked tokens
            masked_predictions = predictions[mask]
            masked_ground_truth = batch[mask]
            
            # Calculate accuracy
            accuracy_mask = (masked_predictions == masked_ground_truth)
            
            if verbose:
                for i in range(len(masked_predictions)):
                    print(f"Predicted: {tokenizer.id_to_token[masked_predictions[i].item()]} | "
                          f"Expected: {tokenizer.id_to_token[masked_ground_truth[i].item()]}")
            
            correct += accuracy_mask.sum().item()
            total += len(masked_predictions)
    
    accuracy = correct / total if total > 0 else 0
    print(f"Mask Prediction Accuracy: {accuracy:.2%}")
    print(f"Correct: {correct} | Total: {total}")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions.json", help="Path to dataset json file")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of captions to evaluate")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="Probability of masking each token")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Length of text embedding vectors")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Units in hidden layers")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    tokenizer.load(args.pkl)
    vocab_size = tokenizer.get_vocab_size()
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    
    if "lstm" in args.model_file.lower():
        model = LSTMModel(vocab_size, embedding_dim, hidden_dim).to(device)
    elif "transformer" in args.model_file.lower():
        model = TransformerModel(vocab_size, embedding_dim, hidden_dim).to(device)
    else:
        raise ValueError("Model type could not be determined from filename.")
    
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    print(f"Loaded model from {args.model_file}")
    
    dataset = LevelDataset(args.json, tokenizer, batch_size=16, mode="mlm")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    evaluate_model(model, tokenizer, dataloader, device, args.mask_prob)
