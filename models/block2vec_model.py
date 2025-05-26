import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from safetensors.torch import save_file, load_file

class Block2Vec(nn.Module):
    """Block2Vec model that learns tile embeddings through context prediction"""
    
    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size (int): Number of unique tiles
            embedding_dim (int): Size of embedding vectors
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Two embedding layers - one for target tiles, one for context tiles
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_ids, context_ids):
        """
        Forward pass computing loss for predicting context tiles given center tile
        
        Args:
            center_ids: Tensor of shape (batch_size,) containing target tile IDs
            context_ids: Tensor of shape (batch_size, context_size) containing context tile IDs
        Returns:
            Tensor containing loss value
        """
        # Flatten context_ids to shape (batch * context_len)
        batch_size, context_len = context_ids.shape
        center_ids_expanded = center_ids.unsqueeze(1).expand(-1, context_len).reshape(-1)
        context_ids_flat = context_ids.reshape(-1)

        center_vec = self.in_embed(center_ids_expanded)  # (batch * context_len, dim)
        context_vec = self.out_embed(context_ids_flat)   # (batch * context_len, dim)

        scores = (center_vec * context_vec).sum(dim=1)  # dot product
        loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores))  # positive pairs

        return loss

    def get_embeddings(self):
        """Returns the learned embeddings for all tiles"""
        return self.in_embed.weight.detach()

    def save_pretrained(self, save_directory):
        """Save model in HuggingFace format"""
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save model weights using safetensors
        save_file(
            self.state_dict(),
            os.path.join(save_directory, "model.safetensors")
        )

        # Save embeddings separately for easy access
        torch.save(
            self.get_embeddings(),
            os.path.join(save_directory, "embeddings.pt")
        )

    @classmethod
    def from_pretrained(cls, model_directory):
        """Load model in HuggingFace format"""
        # Load config
        with open(os.path.join(model_directory, "config.json")) as f:
            config = json.load(f)

        # Initialize model
        model = cls(**config)

        # Load weights
        state_dict = load_file(os.path.join(model_directory, "model.safetensors"))
        model.load_state_dict(state_dict)

        return model