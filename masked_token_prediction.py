import argparse
import torch
import random
from tokenizer import Tokenizer
from models import LSTMModel, TransformerModel
from level_dataset import LevelDataset
from torch.utils.data import DataLoader
from mlm_training import masked_inputs

def evaluate_model(model, tokenizer, dataloader, device, mask_prob=0.15):
    model.eval()
    correct, total = 0, 0
    for batch in dataloader:
        for item in batch:
            masked_input = masked_inputs(item.clone(), tokenizer, device, mask_prob)
            ground_truth = item.clone()
            masked_indices = (masked_input != ground_truth).nonzero().squeeze(1)

            input_tensor = torch.tensor(masked_input).unsqueeze(0).to(device)
            ground_truth_tensor = torch.tensor(ground_truth).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            predicted_ids = output[0].argmax(-1).tolist()
            
            for idx in masked_indices:
                predicted_token = tokenizer.id_to_token[predicted_ids[idx]]
                expected_token = tokenizer.id_to_token[ground_truth_tensor[idx].item()]
                print(f"Original: {' '.join(tokenizer.decode(ground_truth))}")
                print(f"Masked: {' '.join(tokenizer.decode(masked_input))}")
                print(f"Predicted: {predicted_token} | Expected: {expected_token}\n")
                
                if predicted_token == expected_token:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Mask Prediction Accuracy: {accuracy:.2%}")

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
