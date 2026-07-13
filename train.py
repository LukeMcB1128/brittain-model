"""
Training loop for BRITTAIN v2.

Speed / quality upgrades over the original transformer.py:
  * bf16 autocast on MPS (big speedup, M3 Max supports it).
  * Gradient accumulation to reach a large effective batch without OOM.
  * Cosine LR schedule with warmup + gradient clipping (stable, higher LR).
  * Averaged eval loss over several batches (less noisy than one-batch print).
  * Fused attention + RoPE model (see model.py).

Edit the CONFIG block for your run. Pick a size preset from RECOMMENDATIONS.md.
"""
import os
import time
import math
import pickle

import numpy as np
import torch

from model import Brittain, GPTConfig

# ----------------------------- CONFIG -----------------------------
# Size preset (see RECOMMENDATIONS.md). This is the "fast iteration" tier.
block_size   = 512
n_layer      = 8
n_head       = 8
n_embd       = 512
dropout      = 0.1        # some regularization helps on a small dataset

batch_size        = 24    # sequences per micro-step; lower if you hit OOM
grad_accum_steps  = 4     # effective batch = batch_size * grad_accum_steps
max_iters         = 6000
warmup_iters      = 200
learning_rate     = 6e-4
min_lr            = 6e-5
weight_decay      = 0.1
grad_clip         = 1.0

eval_interval = 250
eval_iters    = 50
out_path      = "brittain_v2.pt"
data_dir      = "./data"
resume        = True      # continue from out_path if it exists
# ------------------------------------------------------------------

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"--- device: {device} ---")

with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data   = np.memmap(os.path.join(data_dir, 'val.bin'),   dtype=np.uint16, mode='r')


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout)
model = Brittain(cfg).to(device)
print(f"Parameters: {model.num_params():,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                              betas=(0.9, 0.95), weight_decay=weight_decay)

start_iter = 0
if resume and os.path.exists(out_path):
    ck = torch.load(out_path, map_location=device)
    model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optim'])
    start_iter = ck['iter'] + 1
    print(f"Resumed from {out_path} at iter {start_iter}")


def lr_at(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


print(f"Training {start_iter} -> {max_iters} (effective batch "
      f"{batch_size * grad_accum_steps} seqs x {block_size} tokens)")
t0 = time.time()
for it in range(start_iter, max_iters + 1):
    for g in optimizer.param_groups:
        g['lr'] = lr_at(it)

    optimizer.zero_grad(set_to_none=True)
    for micro in range(grad_accum_steps):
        x, y = get_batch('train')
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(x, y)
            loss = loss / grad_accum_steps
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if it % eval_interval == 0:
        stats = estimate_loss()
        dt = time.time() - t0
        print(f"iter {it:5d} | train {stats['train']:.4f} | val {stats['val']:.4f} "
              f"| lr {lr_at(it):.2e} | {dt:.0f}s")
        torch.save({'iter': it, 'model': model.state_dict(),
                    'optim': optimizer.state_dict(), 'cfg': cfg.__dict__}, out_path)

print("Done. Saved to", out_path)
