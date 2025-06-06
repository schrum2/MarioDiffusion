import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import os
import json
import threading
from util.plotter import Plotter  # Import the Plotter class
from patch_dataset import PatchDataset
import torch.nn.functional as F
from models.block2vec_model_alt import Block2Vec, SkipGramModel
import util.common_settings as common_settings

# ====== Defaults, but overridden by params ======
EMBEDDING_DIM = 16
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
VOCAB_SIZE = common_settings.MARIO_TILE_COUNT 

def print_nearest_neighbors(model, tile_id, k=5):
    emb = model.embeddings.weight
    #emb = model.target_embeddings.weight
    norm_emb = F.normalize(emb, dim=1)
    target = norm_emb[tile_id].unsqueeze(0)
    sims = F.cosine_similarity(target, norm_emb)
    topk = sims.topk(k + 1)  # include itself
    for i in topk.indices[1:]:  # skip self
        print(f"Tile {i.item()} similarity: {sims[i].item():.3f}")

# ====== Training ======
def main():
    parser = argparse.ArgumentParser(description="Train Block2Vec model")
    parser.add_argument('--json_file', type=str, required=True, help='Path to the JSON dataset file')
    parser.add_argument('--output_dir', type=str, default='output', help='Path to the output directory for embeddings')
    parser.add_argument('--embedding_dim', type=int, default=EMBEDDING_DIM, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LR, help='Learning rate')

    args = parser.parse_args()


    # Check if the output directory exists
    if os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' already exists. Please remove it or choose a different name.")
        exit()
    else:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir)

    # Load dataset
    dataset = PatchDataset(json_path=args.json_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Determine vocab size from the dataset
    #vocab_size = max(max(patch) for sample in dataset.patches for patch in sample) + 1

    # Modified voacb size calculation with type conversion
    try: 
        # vocab_size = max(max(patch) for sample in dataset.patches for patch in sample) + 1
        # # Convert to integer if necessary
        # vocab_size = int(vocab_size)
        vocab_size = int(VOCAB_SIZE) # set to vocab size from common settings
    except ValueError as e:
        print(f"Error converting tile IDs to integers: {e}")
        raise
    print(f"Detected vocab size: {vocab_size}")


    # Model, optimizer
    model = Block2Vec(vocab_size=vocab_size, embedding_dim=args.embedding_dim)
    #model = SkipGramModel(emb_size=vocab_size, emb_dimension=args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss() # calculates the error rate between the predicted value and the original value

    # Initialize Plotter
    log_file = os.path.join(args.output_dir, 'training_log.jsonl')
    plotter = Plotter(log_file=log_file, update_interval=5.0, left_key='loss', left_label='Loss', output_png='training_progress.png')

    # Start plotting in a background thread
    plot_thread = threading.Thread(target=plotter.start_plotting)
    plot_thread.daemon = True
    plotter.running = True
    plot_thread.start()

    for epoch in range(args.epochs):
        total_loss = 0
        for center, context in dataloader:
            center_ids_expanded = center.unsqueeze(1).expand(-1, len(context[0]))#.reshape(-1)
            #context_compressed = context.reshape(-1)
            optimizer.zero_grad()
            output = model(context)
            output = output.permute(0, 2, 1)
            #output = output.argmax(dim=2)
            #center_ids_expanded = F.one_hot(center_ids_expanded)
            #print(f"Output size: {output.size()}")
            #print(f"Center size: {center_ids_expanded.size()}")
            loss = loss_function(output, center_ids_expanded)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

        # Log the loss to the log file
        with open(log_file, 'a') as f:
            log_data = {'epoch': epoch + 1, 'loss': total_loss}
            f.write(json.dumps(log_data) + '\n')

        # Update the plot
        plotter.update_plot()

    print("Done: show nearest neighbors of each tile")
    for tile_id in range(vocab_size): 
        print(f"Top neighbors of tile {tile_id}")
        print_nearest_neighbors(model, tile_id, k=5)

    # ====== Save Embeddings ======
    model.save_pretrained(args.output_dir)
    print(f"Embeddings saved to {args.output_dir}")

    # Stop the plotting thread
    plotter.stop_plotting()
    plot_thread.join(timeout=1)

if __name__ == "__main__":
    main()
