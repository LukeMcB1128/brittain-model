"""
Instruction-tune (SFT) the pretrained BRITTAIN-124M base model.

Same training loop as pretraining, four differences (see the RLHF/SFT chat):
  * starts from the BASE checkpoint (brittain_124m_best.pt), not random init
  * tiny data (~50K Alpaca examples), few epochs
  * low learning rate, so it learns to follow instructions without wrecking the
    language ability it already has (catastrophic forgetting)
  * LOSS MASKING via -100 labels (built in prepare_sft.py); the model's
    cross-entropy ignores those positions by default, so only response tokens
    are trained.

Run on the box:  python3 prepare_sft.py ; python3 train_sft.py
Writes brittain_124m_sft.pt.
"""
import os
import time
import math

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from model import Brittain, GPTConfig

BASE = "brittain_124m_best.pt"
OUT = "brittain_124m_sft.pt"
batch_size = 16
epochs = 3
learning_rate = 2e-5          # low: nudge behavior, don't overwrite knowledge
warmup_steps = 100
grad_clip = 1.0
log_interval = 50

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"--- SFT on device: {device} ---")

ck = torch.load(BASE, map_location=device)
cfg = GPTConfig(**ck['cfg'])
model = Brittain(cfg).to(device)
model.load_state_dict(ck['model'])
model.train()
print(f"Loaded base model: {model.num_params():,} params (from {BASE})")

X = np.load("data/sft_input_ids.npy")   # uint16 [N, L]
Y = np.load("data/sft_labels.npy")      # int32  [N, L]  (-100 = ignore)
N = len(X)
steps_per_epoch = N // batch_size
max_steps = epochs * steps_per_epoch
print(f"{N} examples | {steps_per_epoch} steps/epoch | {max_steps} total steps")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                              betas=(0.9, 0.95), weight_decay=0.0)


def lr_at(step):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.1 * learning_rate + 0.5 * (1 + math.cos(math.pi * prog)) * (0.9 * learning_rate)


def get_batch(order, s):
    idx = order[s * batch_size:(s + 1) * batch_size]
    x = torch.from_numpy(X[idx].astype(np.int64))
    y = torch.from_numpy(Y[idx].astype(np.int64))
    # model grades logits[i] against targets[i]; shift so it predicts next token
    return x[:, :-1].to(device), y[:, 1:].to(device)


step = 0
t0 = time.time()
for epoch in range(epochs):
    order = np.random.permutation(N)
    for s in range(steps_per_epoch):
        for g in optimizer.param_groups:
            g['lr'] = lr_at(step)
        x, y = get_batch(order, s)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(x, y)        # -100 labels ignored by cross_entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        step += 1
        if step % log_interval == 0:
            dt = time.time() - t0
            eta = (max_steps - step) * dt / step / 60
            print(f"epoch {epoch+1}/{epochs} | step {step}/{max_steps} | "
                  f"loss {loss.item():.3f} | lr {lr_at(step):.1e} | ETA {eta:.0f} min",
                  flush=True)
    torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__, 'epoch': epoch + 1}, OUT)
    print(f"--> saved {OUT} after epoch {epoch+1}", flush=True)

print("SFT done. ->", OUT)
