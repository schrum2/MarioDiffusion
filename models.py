import torch
import torch.nn as nn
import math

# Transformer model for MLM training

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads=8, num_layers=4, max_seq_length=60):
        super(TransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Proper positional encoding
        self.positional_encoding = self.create_positional_encoding(max_seq_length, embedding_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Projection layer
        self.fc = nn.Linear(embedding_dim, vocab_size)
    
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
        # Ensure positional encoding is on the same device as input
        pe = self.positional_encoding[:, :x.size(1), :].to(x.device)
        
        # Embed input and add positional encoding
        embedded = self.embedding(x) + pe
        
        # Pass through transformer
        transformer_out = self.transformer(embedded)

        return transformer_out
        
    def forward(self, x):
        transformer_out = self.get_embeddings(x)
        # Project to vocabulary size
        output = self.fc(transformer_out)
        return output
