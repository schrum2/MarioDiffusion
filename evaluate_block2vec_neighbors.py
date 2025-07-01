import torch
from create_ascii_captions import extract_tileset
import os
from train_block2vec import print_nearest_neighbors
import util.common_settings as common_settings

def evaluate_block2vec_neighbors(embedding_dir: str):
    """Print embeddings for each tile from a trained block2vec model"""
    
    # Load tileset information
    tileset_path = common_settings.MARIO_TILESET
    tile_chars, id_to_char, _, _ = extract_tileset(tileset_path)
    
    # Load embeddings
    embedding_path = os.path.join(embedding_dir, 'embeddings.pt')
    try:
        embeddings = torch.load(embedding_path)
        print(f"\nLoaded embeddings with shape: {embeddings.shape}")
        
        # Create a mock model to use with print_nearest_neighbors
        class MockModel:
            def __init__(self, embeddings):
                self.in_embed = torch.nn.Embedding.from_pretrained(embeddings)
        
        model = MockModel(embeddings)
        
        # Print embeddings for each tile
        print("\nTile Embeddings:")
        print("-" * 50)
        for tile_id, char in id_to_char.items():
            if tile_id < len(embeddings):
                embedding = embeddings[tile_id]
                print(f"\nTile {tile_id} ('{char}'):")
                print(f"Embedding: {embedding.tolist()}")
                print_nearest_neighbors(model, tile_id, k=5)  # Use the function from train_block2vec
        
    except FileNotFoundError:
        print(f"Error: Could not find embeddings at {embedding_path}")
        print("Make sure the path to your embeddings is correct")

if __name__ == "__main__":
    embedding_dir = "SMB1-block2vec-embeddings"
    evaluate_block2vec_neighbors(embedding_dir)