import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from safetensors.torch import save_file, load_file

class Block2VecModel(nn.Module):
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
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_tiles, context_tiles):
        """
        Forward pass computing loss for predicting context tiles given center tile
        
        Args:
            center_tiles: Tensor of shape (batch_size,) containing target tile IDs
            context_tiles: Tensor of shape (batch_size, context_size) containing context tile IDs
        Returns:
            Tensor containing loss value
        """
        # Get embeddings for center and context tiles
        center_embeds = self.target_embeddings(center_tiles)  # (batch_size, embed_dim)
        context_embeds = self.context_embeddings(context_tiles)  # (batch_size, context_size, embed_dim)
        
        # Compute dot product between center and context embeddings
        logits = torch.einsum('be,bce->bc', center_embeds, context_embeds)
        
        # Loss is negative log softmax 
        log_probs = F.log_softmax(logits, dim=1)
        loss = -log_probs.mean()
        
        return loss

    def get_embeddings(self):
        """Returns the learned embeddings for all tiles"""
        return self.target_embeddings.weight.detach()

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