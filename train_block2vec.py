import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# ====== Config ======
EMBEDDING_DIM = 32
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
VOCAB_SIZE = 16  # number of real tile types (adjust as needed)

# ====== Your dataset class (must yield (center, context_list)) ======
# Dataset must yield:
#   center_tile: int
#   context_tiles: List[int] (length ≤ 8, excluding -1s)
from mario_dataset import MarioPatchDataset

dataset = MarioPatchDataset(json_path='test_tiles.json')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
model = Block2Vec(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0
    for center, context in dataloader:
        # Filter out any context samples that are completely -1
        mask = (context != -1)
        valid_context = []
        valid_center = []
        for c, ctx_row, m in zip(center, context, mask):
            ctx_ids = ctx_row[m].tolist()
            if len(ctx_ids) == 0:
                continue
            valid_context.append(torch.tensor(ctx_ids))
            valid_center.append(c)

        if not valid_center:
            continue

        # Pad all context vectors to max length
        max_len = max(len(c) for c in valid_context)
        padded_context = torch.full((len(valid_context), max_len), fill_value=0, dtype=torch.long)
        for i, ctx in enumerate(valid_context):
            padded_context[i, :len(ctx)] = ctx

        center_tensor = torch.stack(valid_center)

        loss = model(center_tensor, padded_context)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# ====== Save Embeddings ======
torch.save(model.in_embed.weight.detach(), "block2vec_embeddings.pt")
