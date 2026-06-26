import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import os
import json
import threading
from util.plotter import Plotter  # Import the Plotter class
from patch_dataset import PatchDataset
from models.block2vec_model import Block2Vec
import util.common_settings as common_settings

# ====== Defaults, but overridden by params ======
EMBEDDING_DIM = 16
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
NEGATIVE_SAMPLES = 5
VOCAB_SIZE = common_settings.MARIO_TILE_COUNT

def print_nearest_neighbors(model, tile_id, k=5):
    emb = model.in_embed.weight
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
    parser.add_argument('--negative_samples', type=int, default=NEGATIVE_SAMPLES, help='Number of negative context tiles per positive pair')
    parser.add_argument('--vocab_size', type=int, default=None, help='Number of tile types. Defaults to the largest tile id in the data + 1. Set this to the tileset size so every tile id gets an embedding row.')
    parser.add_argument('--use_class_weights', action='store_true', help='Use inverse-frequency class weights to upweight rare center tiles')
    parser.add_argument('--focal_gamma', type=float, default=0.0, help='Focal loss gamma. 0 = disabled')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing (not used for BCE negative sampling, kept for future)')
    parser.add_argument('--save_every', type=int, default=20, help='Save checkpoint every N epochs. 0 disables periodic checkpointing.')
    parser.add_argument('--lr_patience', type=int, default=3,
                        help='Reduce learning rate when training loss has not improved for this many epochs. 0 disables LR scheduling.')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='Factor to reduce the learning rate by when plateauing.')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum learning rate for ReduceLROnPlateau.')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Stop training after this many epochs with no improvement. 0 disables early stopping.')
    parser.add_argument('--early_stop_delta', type=float, default=1e-4,
                        help='Minimum loss improvement to reset early stopping counter.')
    parser.add_argument('--min_epochs', type=int, default=20,
                        help='Minimum number of epochs to run before early stopping can trigger.')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Clip gradient norm to this value. 0 disables clipping.')
    parser.add_argument('--normalize_embeddings', action='store_true',
                        help='Normalize final saved embeddings to unit norm.')

    args = parser.parse_args()


    # Check if the output directory exists
    if os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' already exists. Please remove it or choose a different name.")
        exit()
    else:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir)

    # Load dataset
    dataset = PatchDataset(json_path=args.json_file, output_dir=args.output_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Compute vocab size from the actual dataset to handle any tile set (Mario, MM, etc.)
    try:
        detected_vocab = max(max(patch) for sample in dataset.patches for patch in sample) + 1
        detected_vocab = int(detected_vocab)
    except ValueError as e:
        print(f"Error converting tile IDs to integers: {e}")
        raise

    # An explicit --vocab_size lets us size the embedding table to the full tileset,
    # so every tile id has a row even if it never appears in the sampled windows
    # (otherwise the diffusion model can index past the end of the embeddings).
    if args.vocab_size is not None:
        if args.vocab_size < detected_vocab:
            raise ValueError(f"--vocab_size {args.vocab_size} is smaller than the largest tile id in the data (needs at least {detected_vocab}).")
        vocab_size = args.vocab_size
    else:
        vocab_size = detected_vocab
    print(f"Using vocab size: {vocab_size}")


    # Model, optimizer
    model = Block2Vec(vocab_size=vocab_size, embedding_dim=args.embedding_dim, negative_samples=args.negative_samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            # verbose=True, # deprecated
        )
    best_loss = float('inf')
    epochs_since_improvement = 0

    # Initialize Plotter
    log_file = os.path.join(args.output_dir, 'training_log.jsonl')
    plotter = Plotter(log_file=log_file, update_interval=5.0, left_key='loss', left_label='Loss', output_png='training_progress.png', right_key=None, right_label=None)

    # Start plotting in a background thread
    plot_thread = threading.Thread(target=plotter.start_plotting)
    plot_thread.daemon = True
    plotter.running = True
    plot_thread.start()

    def save_checkpoint(model, output_dir, epoch):
        checkpoint_dir = os.path.join(output_dir, f'checkpoint_epoch{epoch}')
        model.save_pretrained(checkpoint_dir)

    for epoch in range(args.epochs):
        total_loss = 0
        # Per-class accumulators for diagnostics
        per_class_loss_sum = [0.0] * vocab_size
        per_class_count = [0] * vocab_size
        for center, context in dataloader:
            optimizer.zero_grad()

            # If class weights requested, build a per-example weight vector matching expanded centers
            if args.use_class_weights:
                # dataset.center_counts exists and contains counts for centers
                freqs = [dataset.center_counts.get(i, 0) for i in range(vocab_size)]
                freqs = [f if f > 0 else 1 for f in freqs]
                inv_weights = [1.0 / (f ** 0.5) for f in freqs]
                weight_tensor = torch.tensor(inv_weights, dtype=torch.float)
                batch_size, context_len = context.shape
                center_expanded = center.unsqueeze(1).expand(-1, context_len).reshape(-1)
                sample_weights = weight_tensor[center_expanded]
            else:
                sample_weights = None

            # Use the model's new API to return per-example loss so we can aggregate per-class diagnostics
            per_example_loss = model(center, context, sample_weights=sample_weights, focal_gamma=args.focal_gamma, return_per_example=True)
            loss = per_example_loss.mean()
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            total_loss += loss.item()

            # Accumulate per-class stats
            batch_centers = center.unsqueeze(1).expand(-1, context.shape[1]).reshape(-1)
            for i, c in enumerate(batch_centers.tolist()):
                per_class_loss_sum[c] += per_example_loss[i].item()
                per_class_count[c] += 1

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f} lr={current_lr:.6g}")

        # Log the loss to the log file
        with open(log_file, 'a') as f:
            log_data = {'epoch': epoch + 1, 'loss': total_loss, 'lr': current_lr}
            # Add per-class average losses for classes seen this epoch
            per_class_avg = {str(i): per_class_loss_sum[i] / per_class_count[i] if per_class_count[i] > 0 else None for i in range(vocab_size)}
            log_data['per_class_avg_loss'] = per_class_avg
            f.write(json.dumps(log_data) + '\n')

        # Update the plot
        plotter.update_plot()

        if scheduler is not None:
            scheduler.step(total_loss)

        if total_loss + args.early_stop_delta < best_loss:
            best_loss = total_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0 and epoch + 1 < args.epochs:
            save_checkpoint(model, args.output_dir, epoch + 1)

        if args.early_stop_patience > 0 and epoch + 1 >= args.min_epochs and epochs_since_improvement >= args.early_stop_patience:
            print(f"Early stopping after epoch {epoch+1}: no improvement in {epochs_since_improvement} epochs.")
            break

    print("Done: show nearest neighbors of each tile")
    for tile_id in range(vocab_size):
        print(f"Top neighbors of tile {tile_id}")
        print_nearest_neighbors(model, tile_id, k=5)

    if args.normalize_embeddings:
        with torch.no_grad():
            model.in_embed.weight.data = F.normalize(model.in_embed.weight.data, dim=1)
            model.out_embed.weight.data = F.normalize(model.out_embed.weight.data, dim=1)

    # ====== Save Embeddings ======
    model.save_pretrained(args.output_dir)
    print(f"Embeddings saved to {args.output_dir}")

    # Stop the plotting thread
    plotter.stop_plotting()
    plot_thread.join(timeout=1)

if __name__ == "__main__":
    main()
