import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from safetensors.torch import save_file, load_file

class Block2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_ids, context_ids):
        # center_ids: (batch,)
        # context_ids: (batch, context_len)
        center_vec = self.in_embed(center_ids)                 # (batch, dim)
        context_vec = self.out_embed(context_ids)              # (batch, context_len, dim)
        
        # Dot product between center and each context
        score = torch.einsum('bd,bkd->bk', center_vec, context_vec)  # (batch, context_len)
        log_probs = F.log_softmax(score, dim=1)                      # (batch, context_len)
        loss = -log_probs.mean()
        return loss

    def get_embeddings(self):
        """Returns the learned embeddings"""
        return self.in_embed.weight.detach()

    def save_pretrained(self, save_directory):
        """Save model in HuggingFace format"""
        os.makedirs(save_directory, exist_ok=True)

        # Save configuration
        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

        # Save model weights
        save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"))

        # Save embeddings separately for easy access
        torch.save(self.get_embeddings(), os.path.join(save_directory, "embeddings.pt"))

    @classmethod
    def from_pretrained(cls, load_directory):
        """Load model in HuggingFace format"""
        with open(os.path.join(load_directory, "config.json")) as f:
            config = json.load(f)

        model = cls(**config)

        # Load weights
        state_dict = load_file(os.path.join(load_directory, "model.safetensors"))
        model.load_state_dict(state_dict)

        return model