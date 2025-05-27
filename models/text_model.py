import torch
import torch.nn as nn
import math
import os
import json
from safetensors.torch import save_file, load_file
from tokenizer import Tokenizer

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
    def from_pretrained(cls, load_directory):
        with open(os.path.join(load_directory, "config.json")) as f:
            config = json.load(f)

        model = cls(**config)

        # Load weights
        state_dict = load_file(os.path.join(load_directory, "model.safetensors"))
        model.load_state_dict(state_dict)

        # Load tokenizer if available
        tokenizer_path = os.path.join(load_directory, "tokenizer.pkl")
        if os.path.exists(tokenizer_path):
            tokenizer = Tokenizer()
            tokenizer.load(tokenizer_path)
            model.tokenizer = tokenizer

        return model
