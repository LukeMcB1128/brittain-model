"""
Fast, low-cost fine-tune that teaches the model to use context beyond 32
tokens WITHOUT retraining the whole 604M-parameter model from scratch.

Strategy:
- Load the ORIGINAL untouched checkpoint (brittain_model_backup.pt), which
  has a fully-trained 32-slot position embedding table.
- Expand the position embedding table to 256 slots. Slots 0-31 are copied
  from the trained weights (kept as a good starting point); slots 32-255
  start as small random noise (like migrate.py) since they need to be
  learned - tiling was tried and caused attention aliasing.
- FREEZE every other parameter in the model (token embeddings, attention,
  MoE experts, layer norms, lm_head). Only position_embedding_table.weight
  requires gradients.
- Train on real data at block_size=256 for a modest number of iterations.
  Because gradients only flow into ~256*1024 = 262K parameters (vs 604M),
  each step is dominated by the forward/backward pass through the frozen
  network, but the OPTIMIZER only updates a tiny slice of memory, and you
  need far fewer iterations to get a useful signal than training from
  scratch, since the rest of the network already knows how to use whatever
  positional bias it's given.

This won't reach the quality of a full retrain, but it should meaningfully
extend usable context past 32 without an 8-hour run.

Usage:
    python3 finetune_positions.py

Adjust MAX_ITERS / LEARNING_RATE below if you want a longer/shorter run.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import pickle
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"--- Fine-tuning position embeddings on device: {device} ---")

# --- Hyperparameters (must match the trained architecture) ---
OLD_CONTEXT = 32
NEW_CONTEXT = 256
n_embd = 1024
n_head = 8
n_layer = 16
num_experts = 4
batch_size = 16
MAX_ITERS = 400          # small run - only the position table is learning
EVAL_INTERVAL = 50
LEARNING_RATE = 3e-3     # higher LR is fine since so few params are training
BACKUP_PATH = "brittain_model_backup.pt"
OUTPUT_PATH = "brittain_model.pt"
DATA_DIR = "./data"

# --- Load vocab + data ---
with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
itos = meta['itos']
decode = lambda l: ''.join([itos[i] for i in l])

train_data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - NEW_CONTEXT, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+NEW_CONTEXT]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+NEW_CONTEXT+1]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# --- Architecture (identical to transformer.py, block_size=NEW_CONTEXT) ---
block_size = NEW_CONTEXT

class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        wei = self.query(x) @ self.key(x).transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return F.softmax(wei, dim=-1) @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x): return self.net(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, n_embd, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])
        self.router = nn.Linear(n_embd, num_experts)
    def forward(self, x):
        B, T, C = x.shape
        flat_x = x.view(-1, C)
        router_logits = self.router(flat_x)
        weights, selected_experts = torch.topk(F.softmax(router_logits, dim=-1), k=1, dim=-1)
        out = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            token_indices = (selected_experts.squeeze(-1) == i).nonzero().squeeze(-1)
            if token_indices.numel() > 0:
                out[token_indices] = weights[token_indices] * expert(flat_x[token_indices])
        return out.view(B, T, C)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = MixtureOfExperts(n_embd, num_experts=num_experts)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Brittain(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

# --- Load backup checkpoint and expand position table ---
print(f"Loading original trained checkpoint from '{BACKUP_PATH}'...")
checkpoint = torch.load(BACKUP_PATH, map_location="cpu")
state_dict = checkpoint['model_state_dict']

pos_key = "position_embedding_table.weight"
old_pos_weights = state_dict[pos_key]  # [32, 1024]
new_pos_weights = torch.zeros((NEW_CONTEXT, n_embd))
new_pos_weights.normal_(mean=0.0, std=0.02)
new_pos_weights[:OLD_CONTEXT, :] = old_pos_weights
state_dict[pos_key] = new_pos_weights

new_tril = torch.tril(torch.ones(NEW_CONTEXT, NEW_CONTEXT))
for key in list(state_dict.keys()):
    if "sa.heads" in key and "tril" in key:
        state_dict[key] = new_tril

model = Brittain().to(device)
model.load_state_dict(state_dict)
print("--> Loaded weights, expanded position table to 256 rows (0-31 trained, 32-255 fresh).")

# --- Freeze everything except the position embedding table ---
for name, param in model.named_parameters():
    param.requires_grad = (name == "position_embedding_table.weight")

trainable = [p for n, p in model.named_parameters() if p.requires_grad]
frozen_count = sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad)
trainable_count = sum(p.numel() for p in trainable)
print(f"--> Trainable parameters: {trainable_count:,} | Frozen parameters: {frozen_count:,}")

optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)

print(f"Starting position-only fine-tune for {MAX_ITERS} iterations at block_size={NEW_CONTEXT}...")
model.train()
for iteration in range(MAX_ITERS):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iteration % EVAL_INTERVAL == 0:
        print(f"Iter {iteration:4d} | Loss: {loss.item():.4f}")
        model.eval()
        with torch.no_grad():
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            sample = context
            for _ in range(80):
                idx_cond = sample[:, -NEW_CONTEXT:]
                l, _ = model(idx_cond)
                l = l[:, -1, :]
                probs = F.softmax(l, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                sample = torch.cat((sample, nxt), dim=1)
            print("Sample:", decode(sample[0].tolist()).replace("\n", " "))
        model.train()

print("Fine-tune complete. Saving checkpoint...")
model.eval()
torch.save({
    'iteration': checkpoint.get('iteration', 0),
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, OUTPUT_PATH)
print(f"--> Saved fine-tuned checkpoint to '{OUTPUT_PATH}'.")
print("Try generate.py with EFFECTIVE_CONTEXT set higher (e.g. 128 or 256) now.")
