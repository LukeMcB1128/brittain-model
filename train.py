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

# reduce CUDA fragmentation OOMs (must be set before torch initializes CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from model import Brittain, GPTConfig

# ----------------------------- CONFIG -----------------------------
# Pick a preset. 'mac_test' = tiny local sanity run on your folder data.
# 'cloud_124m' = the real from-scratch GPT-2-small-scale run on FineWeb.
PRESET = os.environ.get("BRITTAIN_PRESET", "mac_test")

PRESETS = {
    # ~51M params — fast, for confirming the pipeline works on the Mac.
    "mac_test": dict(
        block_size=1024, n_layer=8, n_head=8, n_embd=512, dropout=0.1,
        batch_size=12, grad_accum_steps=8,
        max_iters=2000, warmup_iters=100, learning_rate=6e-4, min_lr=6e-5,
        eval_interval=200, eval_iters=40, out_path="brittain_mac.pt",
    ),
    # ~124M params (GPT-2 small scale) — the real cloud run on real data.
    "cloud_124m": dict(
        block_size=1024, n_layer=12, n_head=12, n_embd=768, dropout=0.0,
        batch_size=16, grad_accum_steps=32,   # ~500K tokens/step; fits L4 24GB
        max_iters=20000, warmup_iters=700, learning_rate=6e-4, min_lr=6e-5,
        eval_interval=500, eval_iters=100, out_path="brittain_124m.pt",
    ),
}
cfg_run = PRESETS[PRESET]
globals().update(cfg_run)
weight_decay = 0.1
grad_clip = 1.0
data_dir = "./data"
resume = True                       # continue from out_path if it exists
compile_model = True                # torch.compile (big speedup on CUDA)
log_interval = 50                   # heartbeat (loss / tok-s / ETA) every N iters
# ------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"--- preset: {PRESET} | device: {device} ---")

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
raw_model = model               # keep an un-compiled handle for saving
print(f"Parameters: {model.num_params():,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                              betas=(0.9, 0.95), weight_decay=weight_decay)

start_iter = 0
best_val = float("inf")
if resume and os.path.exists(out_path):
    ck = torch.load(out_path, map_location=device)
    raw_model.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optim'])
    start_iter = ck['iter'] + 1
    best_val = ck.get('best_val', float("inf"))
    print(f"Resumed from {out_path} at iter {start_iter} (best_val {best_val:.4f})")

if compile_model and device.type == "cuda":
    try:
        print("Compiling model (one-time, ~1 min)...")
        model = torch.compile(model)   # big speedup on CUDA
    except Exception as e:
        print(f"torch.compile failed ({e}); continuing uncompiled.")


def save_ckpt(path, it, val):
    torch.save({'iter': it, 'model': raw_model.state_dict(),
                'optim': optimizer.state_dict(), 'cfg': cfg.__dict__,
                'best_val': best_val, 'val': val}, path)


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
tokens_per_iter = batch_size * grad_accum_steps * block_size
t0 = time.time()
t_log = t0
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

    if it % log_interval == 0 and it > start_iter:
        now = time.time()
        tps = tokens_per_iter * log_interval / (now - t_log)
        eta_min = (max_iters - it) * (now - t_log) / log_interval / 60
        print(f"iter {it:5d} | loss {loss.item() * grad_accum_steps:.3f} "
              f"| {tps/1e3:.0f}k tok/s | ETA {eta_min:.0f} min", flush=True)
        t_log = now

    if it % eval_interval == 0:
        stats = estimate_loss()
        dt = time.time() - t0
        print(f"iter {it:5d} | train {stats['train']:.4f} | val {stats['val']:.4f} "
              f"| lr {lr_at(it):.2e} | {dt:.0f}s", flush=True)
        # always save latest (for crash-resume); also keep the best-val copy
        save_ckpt(out_path, it, stats['val'])
        if stats['val'] < best_val:
            best_val = stats['val']
            save_ckpt(out_path.replace('.pt', '_best.pt'), it, stats['val'])

print("Done. Best val:", best_val, "| latest saved to", out_path)
