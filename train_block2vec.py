import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import os

# ====== Config ======
EMBEDDING_DIM = 32
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
#VOCAB_SIZE = 16  # number of real tile types (adjust as needed)

# ====== Your dataset class (must yield (center, context_list)) ======
# Dataset must yield:
#   center_tile: int
#   context_tiles: List[int] (length ≤ 8, excluding -1s)
from mario_dataset import MarioPatchDataset

# ====== Model ======
class Block2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
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

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load dataset
    dataset = MarioPatchDataset(json_path=args.json_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Determine vocab size from the dataset
    vocab_size = max(max(patch) for sample in dataset.patches for patch in sample) + 1

    # Model, optimizer
    model = Block2Vec(vocab_size=vocab_size, embedding_dim=args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0
        for center, context in dataloader:
            # Filter out any context samples that are completely -1
            #mask = (context != -1) #This line is not needed
            #valid_context = [] #This line is not needed
            #valid_center = [] #This line is not needed
            #for c, ctx_row, m in zip(center, context, mask): #This line is not needed
            #    ctx_ids = ctx_row[m].tolist() #This line is not needed
            #    if len(ctx_ids) == 0: #This line is not needed
            #        continue #This line is not needed
            #    valid_context.append(torch.tensor(ctx_ids)) #This line is not needed
            #    valid_center.append(c) #This line is not needed

            #if not valid_center: #This line is not needed
            #    continue #This line is not needed

            ## Pad all context vectors to max length #This line is not needed
            #max_len = max(len(c) for c in valid_context) #This line is not needed
            #padded_context = torch.full((len(valid_context), max_len), fill_value=0, dtype=torch.long) #This line is not needed
            #for i, ctx in enumerate(valid_context): #This line is not needed
            #    padded_context[i, :len(ctx)] = ctx #This line is not needed

            #center_tensor = torch.stack(valid_center) #This line is not needed

            loss = model(center, context)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    # ====== Save Embeddings ======
    output_path = os.path.join(args.output_dir, "block2vec_embeddings.pt")
    torch.save(model.in_embed.weight.detach(), output_path)
    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    main()