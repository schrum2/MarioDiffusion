"""
embedding_analysis.py

Drop this file into your project (same folder as train_block2vec.py, or
anywhere on your PYTHONPATH). It adds:

  1. UpdateCounter        - tiny helper to track how many times each tile id
                             was actually used as a center during training
                             (this is NOT the same as dataset frequency --
                             it's how many gradient updates that row got).

  2. analyze_embeddings() - call this once at the end of training. It writes
                             a CSV report, a text summary, and several PNG
                             figures into your output directory.

Nothing here modifies Block2Vec or your dataset class.
"""

import os
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class UpdateCounter:
    """Tracks how many times each tile id appeared as an *expanded center*
    during training (i.e. how many times its in_embed row actually got a
    gradient signal). Call .update(batch_centers_expanded) once per batch.
    """

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.counts = np.zeros(vocab_size, dtype=np.int64)

    def update(self, expanded_center_ids):
        """expanded_center_ids: 1D tensor or list of tile ids (post-expand,
        i.e. same thing you already build as `batch_centers` in your loop)."""
        if torch.is_tensor(expanded_center_ids):
            ids = expanded_center_ids.detach().cpu().numpy()
        else:
            ids = np.asarray(expanded_center_ids)
        np.add.at(self.counts, ids, 1)

    def as_dict(self):
        return {i: int(c) for i, c in enumerate(self.counts)}


def _nearest_neighbor_table(emb_matrix, k=5):
    """emb_matrix: (V, D) numpy array (already detached/cpu).
    Returns list of dicts: tile_id -> top-k neighbor ids + cosine sims."""
    norm = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
    sims = norm @ norm.T  # (V, V) cosine similarity
    vocab_size = emb_matrix.shape[0]
    results = []
    for i in range(vocab_size):
        row = sims[i].copy()
        row[i] = -np.inf  # exclude self
        kk = min(k, vocab_size - 1)
        if kk <= 0:
            results.append({"tile_id": i, "neighbors": []})
            continue
        top_idx = np.argpartition(-row, kk - 1)[:kk]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        neighbors = [{"tile_id": int(j), "cosine_sim": float(row[j])} for j in top_idx]
        results.append({"tile_id": i, "neighbors": neighbors})
    return results, sims


def analyze_embeddings(
    model,
    output_dir,
    update_counter=None,
    dataset_center_counts=None,
    init_in_embed=None,
    init_out_embed=None,
    top_k_neighbors=5,
    report_subdir="embedding_analysis",
):
    """
    Call this once, after training finishes (before or after save_pretrained,
    doesn't matter).

    Args:
        model: your trained Block2Vec instance.
        output_dir: your existing args.output_dir.
        update_counter: an UpdateCounter instance you fed during training.
                        If None, update-count-based diagnostics are skipped.
        dataset_center_counts: optional dict {tile_id: count} of how often
                        each tile appears as a center in the raw dataset
                        (you already compute something like this for
                        --use_class_weights via dataset.center_counts).
                        If provided, lets us compare "available in data" vs
                        "actually trained on".
        init_in_embed: optional (V, D) numpy/torch array -- a copy of
                        model.in_embed.weight BEFORE training started, so we
                        can report how far each row moved. If None, this
                        diagnostic is skipped. See note below on capturing
                        this.
        init_out_embed: same idea, for out_embed (optional).
        top_k_neighbors: how many neighbors to report per tile.
        report_subdir: subfolder of output_dir to write everything into.

    Writes into {output_dir}/{report_subdir}/:
        - per_tile_report.csv
        - per_tile_report.json
        - summary.txt
        - fig_update_count_vs_norm.png
        - fig_update_count_vs_movement.png   (only if init_in_embed given)
        - fig_update_count_bar.png
        - fig_pca_embeddings.png
        - fig_similarity_heatmap.png
        - nearest_neighbors.txt
    """
    out_dir = os.path.join(output_dir, report_subdir)
    os.makedirs(out_dir, exist_ok=True)

    vocab_size = model.vocab_size
    in_emb = model.in_embed.weight.detach().cpu().numpy()  # (V, D)

    update_counts = update_counter.counts if update_counter is not None else None
    dataset_counts = None
    if dataset_center_counts is not None:
        dataset_counts = np.array(
            [dataset_center_counts.get(i, 0) for i in range(vocab_size)], dtype=np.int64
        )

    norms = np.linalg.norm(in_emb, axis=1)

    movement = None
    if init_in_embed is not None:
        if torch.is_tensor(init_in_embed):
            init_arr = init_in_embed.detach().cpu().numpy()
        else:
            init_arr = np.asarray(init_in_embed)
        movement = np.linalg.norm(in_emb - init_arr, axis=1)

    # ---------- nearest neighbors ----------
    nn_table, sim_matrix = _nearest_neighbor_table(in_emb, k=top_k_neighbors)

    # ---------- per-tile CSV / JSON report ----------
    csv_path = os.path.join(out_dir, "per_tile_report.csv")
    json_path = os.path.join(out_dir, "per_tile_report.json")

    fieldnames = ["tile_id", "embedding_norm"]
    if update_counts is not None:
        fieldnames.append("train_update_count")
    if dataset_counts is not None:
        fieldnames.append("dataset_center_count")
    if movement is not None:
        fieldnames.append("distance_from_init")
    fieldnames.append("top1_neighbor_id")
    fieldnames.append("top1_neighbor_cosine_sim")

    rows = []
    for i in range(vocab_size):
        row = {"tile_id": i, "embedding_norm": float(norms[i])}
        if update_counts is not None:
            row["train_update_count"] = int(update_counts[i])
        if dataset_counts is not None:
            row["dataset_center_count"] = int(dataset_counts[i])
        if movement is not None:
            row["distance_from_init"] = float(movement[i])
        neighbors = nn_table[i]["neighbors"]
        if neighbors:
            row["top1_neighbor_id"] = neighbors[0]["tile_id"]
            row["top1_neighbor_cosine_sim"] = round(neighbors[0]["cosine_sim"], 4)
        else:
            row["top1_neighbor_id"] = None
            row["top1_neighbor_cosine_sim"] = None
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump(
            {
                "vocab_size": vocab_size,
                "embedding_dim": model.embedding_dim,
                "per_tile": rows,
                "nearest_neighbors": nn_table,
            },
            f,
            indent=2,
        )

    # ---------- nearest neighbor text dump (human-readable) ----------
    nn_txt_path = os.path.join(out_dir, "nearest_neighbors.txt")
    with open(nn_txt_path, "w") as f:
        for entry in nn_table:
            i = entry["tile_id"]
            extra = ""
            if update_counts is not None:
                extra += f" | updates={int(update_counts[i])}"
            if dataset_counts is not None:
                extra += f" | dataset_count={int(dataset_counts[i])}"
            f.write(f"Tile {i} (norm={norms[i]:.3f}{extra})\n")
            for n in entry["neighbors"]:
                f.write(f"    -> tile {n['tile_id']:>3}  cos_sim={n['cosine_sim']:.3f}\n")
            f.write("\n")

    # ---------- flagging likely-undertrained tiles ----------
    flagged = []
    if update_counts is not None:
        nonzero = update_counts[update_counts > 0]
        median_updates = float(np.median(nonzero)) if len(nonzero) else 0.0
        # heuristic: flag tiles with < 5% of median updates, or zero updates
        threshold = max(1.0, 0.05 * median_updates)
        for i in range(vocab_size):
            if update_counts[i] < threshold:
                flagged.append(i)

    # ---------- text summary ----------
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Block2Vec Embedding Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Vocab size: {vocab_size}\n")
        f.write(f"Embedding dim: {model.embedding_dim}\n\n")

        if update_counts is not None:
            f.write(f"Total training-time center updates: {int(update_counts.sum())}\n")
            f.write(f"Median updates per tile (nonzero only): {median_updates:.1f}\n")
            f.write(f"Min updates: {int(update_counts.min())}  |  Max updates: {int(update_counts.max())}\n")
            ratio = (update_counts.max() / max(update_counts.min(), 1))
            f.write(f"Max/min update ratio: {ratio:.1f}x\n\n")

            f.write(f"Tiles flagged as likely UNDERTRAINED (< {threshold:.1f} updates,\n")
            f.write(f"i.e. <5% of median, or zero updates):\n")
            if flagged:
                for i in flagged:
                    f.write(
                        f"  - Tile {i}: {int(update_counts[i])} updates, "
                        f"embedding_norm={norms[i]:.3f}"
                    )
                    if movement is not None:
                        f.write(f", distance_from_init={movement[i]:.4f}")
                    f.write("\n")
            else:
                f.write("  (none flagged)\n")
            f.write("\n")
        else:
            f.write("No update_counter was provided -- skipping update-count diagnostics.\n")
            f.write("(See instructions for hooking UpdateCounter into your training loop.)\n\n")

        if movement is not None:
            f.write("Embedding movement from initialization:\n")
            f.write(f"  Mean distance moved: {movement.mean():.4f}\n")
            f.write(f"  Min distance moved: {movement.min():.4f} (tile {int(np.argmin(movement))})\n")
            f.write(f"  Max distance moved: {movement.max():.4f} (tile {int(np.argmax(movement))})\n\n")
            low_movers = np.where(movement < 0.05 * movement.mean())[0]
            if len(low_movers):
                f.write("Tiles that barely moved from their random init (<5% of mean movement):\n")
                for i in low_movers:
                    f.write(f"  - Tile {int(i)}: moved {movement[i]:.4f}\n")
                f.write("\n")
        else:
            f.write("No init_in_embed snapshot was provided -- skipping 'distance moved' diagnostics.\n")
            f.write("(See instructions for capturing a pre-training snapshot.)\n\n")

        f.write("Nearest-neighbor sanity check (top-1 neighbor per tile):\n")
        for i in range(vocab_size):
            n0 = nn_table[i]["neighbors"][0] if nn_table[i]["neighbors"] else None
            if n0 is not None:
                f.write(f"  Tile {i:>3} -> nearest: tile {n0['tile_id']:>3} (cos_sim={n0['cosine_sim']:.3f})\n")

    # ---------- figures ----------
    # 1. update count vs embedding norm
    if update_counts is not None:
        plt.figure(figsize=(7, 5))
        plt.scatter(update_counts, norms, alpha=0.7)
        for i in range(vocab_size):
            plt.annotate(str(i), (update_counts[i], norms[i]), fontsize=6, alpha=0.6)
        plt.xlabel("Training update count (log scale)")
        plt.ylabel("Embedding L2 norm")
        plt.xscale("symlog")
        plt.title("Update count vs. embedding norm (per tile)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_update_count_vs_norm.png"), dpi=150)
        plt.close()

        # 2. bar chart of update counts (sorted) to visualize imbalance
        order = np.argsort(update_counts)[::-1]
        plt.figure(figsize=(max(7, vocab_size * 0.15), 5))
        plt.bar(range(vocab_size), update_counts[order])
        plt.xticks(range(vocab_size), order, rotation=90, fontsize=6)
        plt.ylabel("Training update count")
        plt.xlabel("Tile id (sorted by update count, descending)")
        plt.yscale("log")
        plt.title("Per-tile training update counts (log scale)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_update_count_bar.png"), dpi=150)
        plt.close()

    # 3. update count vs distance moved from init
    if update_counts is not None and movement is not None:
        plt.figure(figsize=(7, 5))
        plt.scatter(update_counts, movement, alpha=0.7)
        for i in range(vocab_size):
            plt.annotate(str(i), (update_counts[i], movement[i]), fontsize=6, alpha=0.6)
        plt.xlabel("Training update count (log scale)")
        plt.ylabel("Distance moved from init (L2)")
        plt.xscale("symlog")
        plt.title("Update count vs. movement from initialization")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_update_count_vs_movement.png"), dpi=150)
        plt.close()

    # 4. PCA projection of embeddings, colored by update count (or norm if no counts)
    if vocab_size >= 3:
        n_components = 2
        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(in_emb)
        color_vals = update_counts if update_counts is not None else norms
        color_label = "Training update count" if update_counts is not None else "Embedding norm"

        plt.figure(figsize=(7, 6))
        sca = plt.scatter(
            proj[:, 0], proj[:, 1],
            c=np.log1p(color_vals) if update_counts is not None else color_vals,
            cmap="viridis", s=60
        )
        for i in range(vocab_size):
            plt.annotate(str(i), (proj[i, 0], proj[i, 1]), fontsize=7)
        cbar = plt.colorbar(sca)
        cbar.set_label(f"{color_label}" + (" (log1p)" if update_counts is not None else ""))
        var_explained = pca.explained_variance_ratio_
        plt.xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)")
        plt.title("PCA projection of tile embeddings")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_pca_embeddings.png"), dpi=150)
        plt.close()

    # 5. similarity heatmap
    plt.figure(figsize=(max(6, vocab_size * 0.15), max(5, vocab_size * 0.15)))
    plt.imshow(sim_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine similarity")
    plt.xticks(range(vocab_size), range(vocab_size), rotation=90, fontsize=6)
    plt.yticks(range(vocab_size), range(vocab_size), fontsize=6)
    plt.title("Tile embedding cosine similarity matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_similarity_heatmap.png"), dpi=150)
    plt.close()

    print(f"[embedding_analysis] Wrote report + figures to: {out_dir}")
    return out_dir
