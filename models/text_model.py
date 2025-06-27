import argparse
import torch
import torch.nn as nn
import math
import os
import json
from safetensors.torch import save_file, load_file
from tokenizer import Tokenizer
from huggingface_hub import hf_hub_download

def get_embeddings(batch_size, tokenizer, text_encoder, captions=None, neg_captions=None, device='cpu'):
    max_length = text_encoder.max_seq_length
    empty_ids = encode_token_captions([""] * batch_size, tokenizer, max_length, device=device)
    embeddings = text_encoder.get_embeddings(empty_ids)

    if(captions is not None):
        caption_ids = encode_token_captions(captions, tokenizer, max_length, device=device)
        caption_embeddings = text_encoder.get_embeddings(caption_ids)
        embeddings = torch.cat((embeddings, caption_embeddings), dim=0)
    
    if(neg_captions is not None):
        neg_ids = encode_token_captions(neg_captions, tokenizer, max_length, device=device)
        neg_embeddings = text_encoder.get_embeddings(neg_ids)
        embeddings = torch.cat((neg_embeddings, embeddings), dim=0)
    
    return embeddings.to(device)

def encode_token_captions(captions, tokenizer, max_length, device='cpu'):
    caption_ids = []
    for caption in captions:
        tokens = tokenizer.encode(caption)
        caption_tokens = tokenizer.pad_sequence(tokens, max_length)
        caption_ids.append(torch.tensor(caption_tokens, dtype=torch.long).unsqueeze(0))
    return torch.cat(caption_ids, dim=0).to(device)









# Transformer model for MLM training

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tokenizer=None, num_heads=8, num_layers=4, max_seq_length=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self.create_positional_encoding(max_seq_length, embedding_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

        self.tokenizer = tokenizer

    def create_positional_encoding(self, max_seq_length, embedding_dim):
        # The implementation uses a sinusoidal positional encoding, which creates a unique pattern for each position in the sequence.
        # The frequencies create unique values, the sin/cos bounds values
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # Creates a set of divisors that create different frequencies
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_seq_length, embedding_dim)
        # Even dimensions use sin, odd dimensions use cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def get_embeddings(self, x):
        """ This gets the actual latent embedding vectors """
        # Ensure positional encoding is on the same device as input
        pe = self.positional_encoding[:, :x.size(1), :].to(x.device)
        # Embed input and add positional encoding
        embedded = self.embedding(x) + pe
        return self.transformer(embedded)

    def forward(self, x):
        """ This gets the token within the vocabulary """
        transformer_out = self.get_embeddings(x)
        # Project to vocabulary size
        return self.fc(transformer_out)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_seq_length": self.max_seq_length,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

        # Save model weights
        save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"))

        # Save tokenizer if present
        if self.tokenizer is not None:
            self.tokenizer.save(os.path.join(save_directory, "tokenizer.pkl"))

    @classmethod
    def from_pretrained(cls, load_directory, subfolder=None):
        """
        Load a TransformerModel from a local directory or a Hugging Face Hub repo.
        If load_directory is a local path, loads from disk. If not, loads from Hugging Face Hub.
        The subfolder argument specifies a subdirectory (local or in the repo).
        """
        # Helper to get file path (local or HF Hub)
        def get_file(filename):
            # Try local path first
            if os.path.isdir(load_directory):
                model_dir = os.path.join(load_directory, subfolder) if subfolder else load_directory
                file_path = os.path.join(model_dir, filename)
                if os.path.exists(file_path):
                    return file_path
            # Otherwise, try Hugging Face Hub
            repo_id = load_directory
            subpath = f"{subfolder}/{filename}" if subfolder else filename
            return hf_hub_download(repo_id=repo_id, filename=subpath)

        # Load config
        config_path = get_file("config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)

        # Load weights
        weights_path = get_file("model.safetensors")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)

        # Load tokenizer if available
        tokenizer_path = get_file("tokenizer.pkl")
        if tokenizer_path and os.path.exists(tokenizer_path):
            tokenizer = Tokenizer()
            tokenizer.load(tokenizer_path)
            model.tokenizer = tokenizer

        return model
    
    def print_architecture(self, inputs=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, required=True, help="Path to trained transformer model")
        parser.add_argument("--json", type=str, default="SMB1_LevelsAndCaptions-regular-test.json", help="Path to dataset json file")
        parser.add_argument("--num_samples", type=int, default=10, help="Number of captions to evaluate")
        parser.add_argument("--mask_prob", type=float, default=0.15, help="Probability of masking each token")

        parser.add_argument("--compare_checkpoints", action="store_true", default=False, help="Run comparison across all model checkpoints")
        args = parser.parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransformerModel.from_pretrained(args.model_path).to(device)
        print(f"Loaded model from {args.model_path}")

        import os
        import re
        import json
        import matplotlib.pyplot as plt
        from torchview import draw_graph
        import graphviz

        graph = draw_graph(
            model=model,
            input_data=inputs,
            expand_nested=False,
            #enable_output_shape=True,   
            #roll_out="nested",
            depth=1
        )

        # Save plot
        filename = 'mlm_architecture'
        graph.visual_graph.render(filename, format='pdf', cleanup=False)  # Cleanup removes intermediate files
        #graph.visual_graph.save('unet_architecture.dot')

    def save_architecture_pdf(self, filename="transformer_architecture.pdf", input_length=32):
        """Save a visualization of the model architecture as a PDF using torchview."""
        try:
            from torchview import draw_graph
        except ImportError:
            raise ImportError("torchview is required for model visualization. Install with 'pip install torchview'.")
        import torch
        import os
        # Create a dummy input of the correct type for the model
        captions = ["full floor. two coins. one pipe.", "floor with two gaps. one cannon. many enemies."]
        tensor = encode_token_captions(captions, self.tokenizer, self.max_seq_length, device=next(self.parameters()).device)
        input_length = tensor.size(1) if tensor.dim() > 1 else self.max_seq_length

        num_tokens_list = [len(self.tokenizer.encode(c)) for c in captions]
        input_length = max(num_tokens_list) if num_tokens_list else input_length
        dummy_input = torch.zeros((1, input_length), dtype=torch.long, device=next(self.parameters()).device)

        # Draw the graph and save as PNG
        graph = draw_graph(self, input_data=dummy_input, expand_nested=True, save_graph=True, filename=filename.replace('.pdf',''), directory=".", depth=2)
        png_file = filename.replace('.pdf', '.png')
        # Convert PNG to PDF
        if os.path.exists(png_file):
            try:
                from PIL import Image
                im = Image.open(png_file)
                im.save(filename, "PDF", resolution=100.0)
                print(f"Saved architecture PDF to {filename}")
                # Optionally, remove the PNG file
                os.remove(png_file)
            except ImportError:
                print(f"PIL not installed. Architecture saved as PNG: {png_file}")
            except Exception as e:
                print(f"Could not convert PNG to PDF: {e}")
        else:
            print(f"Could not find PNG file to convert: {png_file}")