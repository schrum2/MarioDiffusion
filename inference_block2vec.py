import torch
import argparse
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Model definition must match training
class Block2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = torch.nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_ids, context_ids):
        center_vec = self.in_embed(center_ids)
        context_vec = self.out_embed(context_ids)
        score = torch.einsum('bd,bkd->bk', center_vec, context_vec)
        log_probs = torch.nn.functional.log_softmax(score, dim=1)
        loss = -log_probs.mean()
        return loss

def load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No embedding file found at {path}")
    return torch.load(path)

def find_similar(tile_id, embeddings, top_k=5):
    if tile_id >= embeddings.shape[0]:
        raise ValueError(f"Tile ID {tile_id} exceeds vocab size {embeddings.shape[0]}")
    tile_vec = embeddings[tile_id].unsqueeze(0)
    sims = cosine_similarity(tile_vec, embeddings)[0]
    top = np.argsort(sims)[::-1]
    return [(i, sims[i]) for i in top if i != tile_id][:top_k]

def main():
    parser = argparse.ArgumentParser(description="Infer tile similarity using Block2Vec embeddings")
    parser.add_argument('--embedding_path', type=str, required=True,
                        help='Path to saved block2vec_embeddings.pt')
    parser.add_argument('--tile_id', type=int, required=True,
                        help='Tile ID to query for similar tiles')
    parser.add_argument('--top_k', type=int, default=5, help='Number of similar tiles to return')
    args = parser.parse_args()

    embeddings = load_embeddings(args.embedding_path)
    print(f"Loaded embeddings with shape: {embeddings.shape}")

    similar = find_similar(args.tile_id, embeddings, top_k=args.top_k)

    print(f"\nMost similar tiles to tile {args.tile_id}:")
    for idx, sim in similar:
        print(f"Tile {idx}: Cosine similarity = {sim:.4f}")

if __name__ == "__main__":
    main()
