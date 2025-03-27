import argparse
import torch
from tokenizer import Tokenizer
from models import LSTMModel, TransformerModel

def predict_masked(model, tokenizer, text, device):
    model.eval()
    tokens = tokenizer.encode(text)
    masked_index = tokens.index(tokenizer.token_to_id["[MASK]"]) if "[MASK]" in text else None
    input_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    
    if masked_index is not None:
        predicted_token_id = output[0, masked_index].argmax(-1).item()
        predicted_token = tokenizer.id_to_token[predicted_token_id]
        print(f"Original: {text}")
        print(f"Predicted: {text.replace('[MASK]', predicted_token)}")
    else:
        print("No [MASK] token found in input.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument("--pkl", type=str, default="SMB1_Tokenizer.pkl", help="Path to tokenizer pkl file")
    parser.add_argument("--text", type=str, required=True, help="Input text containing [MASK] token")
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

    predict_masked(model, tokenizer, args.text, device)
