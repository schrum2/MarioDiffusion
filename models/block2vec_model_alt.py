import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from safetensors.torch import save_file, load_file
import time

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
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_ids):
        """
        Forward pass computing output for predicting context tiles given center tile
        
        Args:
            context_ids: Tensor of shape (batch_size, context_size) containing context tile IDs
        Returns:
            The output of the model
        """
        center_vec = self.embeddings(context_ids)  # (batch * context_len, dim)
        output = self.linear(center_vec)   # (batch * context_len, dim)
        return output

    def get_embeddings(self):
        """Returns the learned embeddings for all tiles"""
        return self.embeddings.weight.detach()

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



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SkipGramModel(nn.Module):
    def __init__(self, emb_size: int, emb_dimension: int):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.target_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.output = nn.Linear(emb_dimension, emb_size)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.target_embeddings.weight.data, -
                      initrange, initrange)

    def forward(self, target, context):
        emb_target = self.target_embeddings(target)

        score = self.output(emb_target)
        score = F.log_softmax(score, dim=-1)

        losses = torch.stack([F.nll_loss(score, context_word)
                              for context_word in context.transpose(0, 1)])
        return losses.mean()