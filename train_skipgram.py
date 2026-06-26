import argparse
import os
import sys
import json
import threading
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset
from util.plotter import Plotter

# Ensure repo root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from patch_dataset import PatchDataset


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        initrange = 0.5 / embedding_dim
        nn.init.uniform_(self.in_embed.weight.data, -initrange, initrange)
        nn.init.constant_(self.out_embed.weight.data, 0)

    def forward(self, center_ids, context_ids, negative_ids):
        # center_ids: (B,), context_ids: (B,), negative_ids: (B, neg)
        center_vec = self.in_embed(center_ids)  # (B, D)
        context_vec = self.out_embed(context_ids)  # (B, D)
        pos_scores = (center_vec * context_vec).sum(dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores), reduction='none')

        neg_vec = self.out_embed(negative_ids)  # (B, neg, D)
        neg_scores = (center_vec.unsqueeze(1) * neg_vec).sum(dim=2)  # (B, neg)
        neg_bce = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores), reduction='none')  # (B, neg)
        neg_loss = neg_bce.mean(dim=1)  # mean over negative samples per example

        per_example = pos_loss + neg_loss
        return per_example.mean()


def build_pairs(dataset):
    centers = []
    contexts = []
    for center, context_list in dataset.samples:
        for c in context_list:
            centers.append(center)
            contexts.append(c)
    return np.array(centers, dtype=np.int32), np.array(contexts, dtype=np.int32)


def save_checkpoint(model, output_dir, epoch, vocab_size, embedding_dim, negative_samples):
    checkpoint_dir = os.path.join(output_dir, f'checkpoint_epoch{epoch}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_file(model.state_dict(), os.path.join(checkpoint_dir, 'model.safetensors'))
    torch.save(model.in_embed.weight.detach().cpu(), os.path.join(checkpoint_dir, 'embeddings.pt'))
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump({
            'vocab_size': int(vocab_size),
            'embedding_dim': int(embedding_dim),
            'negative_samples': int(negative_samples)
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', required=True)
    parser.add_argument('--output_dir', default='skipgram_out')
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--negative_samples', type=int, default=10)
    parser.add_argument('--subsampling', action='store_true')
    parser.add_argument('--subsample_threshold', type=float, default=0.001)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--save_every', type=int, default=20,
                        help='Save checkpoint every N epochs. 0 disables periodic checkpointing.')
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

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = PatchDataset(json_path=args.json_file, subsampling=args.subsampling, subsample_threshold=args.subsample_threshold, output_dir=args.output_dir)

    # Determine vocab size
    detected_vocab = max(max(patch) for sample in dataset.patches for patch in sample) + 1
    if args.vocab_size is not None:
        vocab_size = args.vocab_size
    else:
        vocab_size = int(detected_vocab)

    print(f"Building pairs from dataset ({len(dataset.samples)} patches)")
    centers_arr, contexts_arr = build_pairs(dataset)
    print(f"Total positive pairs: {len(centers_arr)}")

    # compute unigram distribution (use center frequencies)
    center_counts = np.zeros(vocab_size, dtype=np.float64)
    for k,v in dataset.center_counts.items():
        if k < vocab_size:
            center_counts[k] = v
    # fallback: +1 smoothing
    center_counts = center_counts + 1e-8
    unigram = center_counts ** 0.75
    unigram = unigram / unigram.sum()

    # Create torch dataset
    centers_t = torch.from_numpy(centers_arr).long()
    contexts_t = torch.from_numpy(contexts_arr).long()
    ds = TensorDataset(centers_t, contexts_t)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=args.embedding_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            # verbose=True, # deprecated
        )
    best_loss = float('inf')
    epochs_since_improvement = 0

    log_file = os.path.join(args.output_dir, 'training_log.jsonl')
    with open(log_file, 'w') as f:
        pass

    plotter = Plotter(log_file=log_file, update_interval=5.0, left_key='loss', left_label='Loss', output_png='training_progress.png', right_key=None, right_label=None)
    plot_thread = threading.Thread(target=plotter.start_plotting)
    plot_thread.daemon = True
    plotter.running = True
    plot_thread.start()

    for epoch in range(args.epochs):
        total_loss = 0.0
        for centers_batch, contexts_batch in loader:
            centers_batch = centers_batch.to(device)
            contexts_batch = contexts_batch.to(device)
            B = centers_batch.size(0)
            # negative sampling
            neg = np.random.choice(vocab_size, size=(B, args.negative_samples), p=unigram)
            neg_t = torch.from_numpy(neg).long().to(device)

            opt.zero_grad()
            loss = model(centers_batch, contexts_batch, neg_t)
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            opt.step()
            total_loss += loss.item()

        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f} lr={current_lr:.6g}")
        with open(log_file, 'a') as f:
            log_data = {'epoch': epoch + 1, 'loss': total_loss, 'lr': current_lr}
            f.write(json.dumps(log_data) + '\n')
        plotter.update_plot()

        if scheduler is not None:
            scheduler.step(total_loss)

        if total_loss + args.early_stop_delta < best_loss:
            best_loss = total_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0 and epoch + 1 < args.epochs:
            save_checkpoint(model, args.output_dir, epoch + 1,
                            vocab_size, args.embedding_dim, args.negative_samples)

        if args.early_stop_patience > 0 and epoch + 1 >= args.min_epochs and epochs_since_improvement >= args.early_stop_patience:
            print(f"Early stopping after epoch {epoch+1}: no improvement in {epochs_since_improvement} epochs.")
            break

    # Save final embeddings
    if args.normalize_embeddings:
        with torch.no_grad():
            model.in_embed.weight.data = F.normalize(model.in_embed.weight.data, dim=1)
            model.out_embed.weight.data = F.normalize(model.out_embed.weight.data, dim=1)

    emb = model.in_embed.weight.detach().cpu()
    torch.save(emb, os.path.join(args.output_dir, 'embeddings.pt'))
    save_file(
        model.state_dict(),
        os.path.join(args.output_dir, 'model.safetensors')
    )
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump({
            'vocab_size': int(vocab_size),
            'embedding_dim': int(args.embedding_dim),
            'negative_samples': int(args.negative_samples)
        }, f, indent=2)

    plotter.stop_plotting()
    plot_thread.join(timeout=1)

    print('Training complete. Embeddings saved to', args.output_dir)


if __name__ == '__main__':
    main()
