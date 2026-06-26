import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from safetensors.torch import save_file, load_file

class Block2Vec(nn.Module):
    """Block2Vec model that learns tile embeddings through context prediction"""

    def __init__(self, vocab_size, embedding_dim, negative_samples=5):
        """
        Args:
            vocab_size (int): Number of unique tiles
            embedding_dim (int): Size of embedding vectors
            negative_samples (int): Number of negative context tiles per positive pair
        """
        super().__init__()
        if negative_samples < 1:
            raise ValueError("negative_samples must be >= 1")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples

        # Two embedding layers - one for target tiles, one for context tiles
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        initrange = 0.5 / embedding_dim
        nn.init.uniform_(self.in_embed.weight.data, -initrange, initrange)
        nn.init.constant_(self.out_embed.weight.data, 0)

    def forward(self, center_ids, context_ids, sample_weights=None, focal_gamma: float = 0.0, return_per_example: bool = False):
        """
        Forward pass computing loss for predicting context tiles given center tile

        Args:
            center_ids: Tensor of shape (batch_size) containing target tile IDs
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

        # Positive pairs: center with its actual context
        pos_scores = (center_vec * context_vec).sum(dim=1)  # (N,)
        pos_targets = torch.ones_like(pos_scores)
        pos_bce = F.binary_cross_entropy_with_logits(pos_scores, pos_targets, reduction='none')

        # Negative sampling: center with noise tiles that are not the positive context tile
        neg_ids = self._sample_negative_ids(context_ids_flat)
        neg_vec = self.out_embed(neg_ids)  # (N, neg_samples, dim)
        neg_scores = (center_vec.unsqueeze(1) * neg_vec).sum(dim=2)  # (N, neg_samples)
        neg_targets = torch.zeros_like(neg_scores)
        neg_bce = F.binary_cross_entropy_with_logits(neg_scores, neg_targets, reduction='none')  # (N, neg_samples)
        # Aggregate negative loss per example (mean over negative samples)
        neg_bce_mean = neg_bce.mean(dim=1)

        # Combine per-example loss
        per_example_loss = pos_bce + neg_bce_mean  # (N,)

        # Apply focal scaling if requested
        if focal_gamma and focal_gamma > 0.0:
            # For focal, compute p_t for positives and negatives separately
            prob_pos = torch.sigmoid(pos_scores)
            prob_neg = torch.sigmoid(neg_scores)
            p_t_pos = prob_pos
            p_t_neg = 1.0 - prob_neg
            # focal weights
            focal_pos = (1.0 - p_t_pos).pow(focal_gamma)
            focal_neg = (1.0 - p_t_neg).pow(focal_gamma)
            focal_neg_mean = focal_neg.mean(dim=1)
            per_example_loss = (pos_bce * focal_pos) + (neg_bce_mean * focal_neg_mean)

        # Apply sample weights if provided (one weight per expanded center id)
        if sample_weights is not None:
            # sample_weights expected shape (N,) matching center_ids_expanded
            per_example_loss = per_example_loss * sample_weights.view(-1).to(per_example_loss.device)

        if return_per_example:
            return per_example_loss

        return per_example_loss.mean()

    def _sample_negative_ids(self, positive_context_ids):
        """Sample negatives for each pair, excluding that pair's true context tile."""
        if self.vocab_size < 2:
            raise ValueError("Negative sampling requires vocab_size >= 2")

        shape = (positive_context_ids.numel(), self.negative_samples)
        neg_ids = torch.randint(0, self.vocab_size, shape, device=positive_context_ids.device)
        positives = positive_context_ids.unsqueeze(1)
        matches_positive = neg_ids.eq(positives)

        while matches_positive.any():
            neg_ids[matches_positive] = torch.randint(
                0,
                self.vocab_size,
                (matches_positive.sum().item(),),
                device=positive_context_ids.device,
            )
            matches_positive = neg_ids.eq(positives)

        return neg_ids

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
            "negative_samples": self.negative_samples,
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
