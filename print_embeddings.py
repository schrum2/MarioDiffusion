import torch
from create_ascii_captions import extract_tileset
import os

def print_tile_embeddings(embedding_dir: str):
    """Print embeddings for each tile from a trained block2vec model"""
    
    # Load tileset information
    tileset_path = os.path.join('..', 'TheVGLC', 'Super Mario Bros', 'smb.json')
    tile_chars, id_to_char, _, _ = extract_tileset(tileset_path)
    
    # Load embeddings
    embedding_path = os.path.join(embedding_dir, 'embeddings.pt')
    try:
        embeddings = torch.load(embedding_path)
        print(f"\nLoaded embeddings with shape: {embeddings.shape}")
        
        # Print embeddings for each tile
        print("\nTile Embeddings:")
        print("-" * 50)
        for tile_id, char in id_to_char.items():
            if tile_id < len(embeddings):
                embedding = embeddings[tile_id]
                print(f"\nTile {tile_id} ('{char}'):")
                print(f"Embedding: {embedding.tolist()}")
        
    except FileNotFoundError:
        print(f"Error: Could not find embeddings at {embedding_path}")
        print("Make sure the path to your embeddings is correct")

if __name__ == "__main__":
    embedding_dir = "SMB1-block2vec-embeddings"
    print_tile_embeddings(embedding_dir)