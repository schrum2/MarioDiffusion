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
    # Move input batch to the correct device
    input_batch = input_batch.to(device)
    # Apply mask only to non-[PAD] tokens
    mask = (torch.rand(input_batch.shape, device=device) < mask_prob) & (input_batch != pad_token)
    input_batch[mask] = mask_token
    return input_batch

def evaluate_model(model, tokenizer, dataloader, device, mask_prob=0.15):
    model.eval()
    mask_token = tokenizer.token_to_id["[MASK]"]
    pad_token = tokenizer.token_to_id["[PAD]"]
    correct, total = 0, 0
    for batch in dataloader:
        for item in batch:
            masked_input = masked_inputs(item.clone(), tokenizer, device, mask_prob)
            ground_truth = item.clone()
            masked_indices = (masked_input == mask_token).nonzero().squeeze(1)

            input_tensor = masked_input.unsqueeze(0).to(device)
            ground_truth_tensor = ground_truth.to(device)

            with torch.no_grad():
                output = model(input_tensor)
            
            predicted_ids = output[0].argmax(-1).tolist()
            
            for idx in masked_indices:
                predicted_token = tokenizer.id_to_token[predicted_ids[idx]]
                expected_token = tokenizer.id_to_token[ground_truth_tensor[idx].item()]
                
                if expected_token == "[PAD]":
                    continue # Don't investigate these

                try:
                    pad_index = ground_truth.tolist().index(pad_token) 
                except ValueError:
                    pad_index = len(ground_truth)

                print(f"Original: {(tokenizer.decode(ground_truth.tolist()[:pad_index]))}")
                print(f"Masked  : {(tokenizer.decode(masked_input.tolist()[:pad_index]))}")
                print(f"Predicted: {predicted_token} | Expected: {expected_token}\n")
                
                if predicted_token == expected_token:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Mask Prediction Accuracy: {accuracy:.2%}")
    print(f"Correct: {correct} | Total: {total}")

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
