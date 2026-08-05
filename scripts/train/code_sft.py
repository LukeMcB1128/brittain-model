"""
Instruction-tune a BRITTAIN-2 coder on the code SFT mix.

The BRITTAIN-2 counterpart to sft.py, which stays as-is because it is what
`brittain1:124m-instruct` was built from. Differences that matter:

  * --base is required, so the run records which checkpoint it came from.
    sft.py hardcoded the 124M path; this chain has four plausible bases.
  * a HELD-OUT SPLIT. sft.py had none, so "3 epochs" was a guess with no way to
    see overfitting. At 33k examples against a 235M model that is a real risk,
    and the val curve is what tells you which epoch to keep.
  * the checkpoint is written in the same shape the rest of the codebase reads —
    cfg, tokenizer and tokenizer_path — so serve.py and sample.py can load it.
    sft.py wrote only {model, cfg, epoch}, which load_tokenizer cannot route.

The prompt format comes from brittain.prompts, shared with chat.py, so what the
model is trained on and what it is prompted with cannot diverge.

    python3 scripts/prepare/prepare_code_sft.py
    python3 scripts/train/code_sft.py --base checkpoints/brittain2_235m_2k_final_weights.pt
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from brittain.model import Brittain, GPTConfig
from brittain.paths import CHECKPOINT_DIR, FIM_TOKENIZER, PROCESSED_DATA_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", default=str(CHECKPOINT_DIR / "brittain2_235m_instruct.pt"))
ap.add_argument("--data_dir", default=str(PROCESSED_DATA_DIR / "code_sft"))
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch_size", type=int, default=16)
# 2e-5, as in sft.py: high enough to move behaviour, low enough not to wash out
# the pretrained code ability. This is a nudge, not a training run.
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--val_frac", type=float, default=0.02)
ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--log_interval", type=int, default=50)
ap.add_argument("--seed", type=int, default=1337)
args = ap.parse_args()

Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"--- code SFT on {device} ---")

ck = torch.load(args.base, map_location="cpu", weights_only=False)
cfg = GPTConfig(**ck["cfg"])
model = Brittain(cfg).to(device)
model.load_state_dict(ck["model"])
model.train()
print(f"Loaded {args.base}: {model.num_params():,} params | ctx {cfg.block_size} "
      f"| vocab {cfg.vocab_size}")

X = np.load(os.path.join(args.data_dir, "input_ids.npy"))
Y = np.load(os.path.join(args.data_dir, "labels.npy"))
with open(os.path.join(args.data_dir, "meta.json")) as f:
    meta = json.load(f)
if meta["vocab_size"] != cfg.vocab_size:
    raise SystemExit(f"data vocab {meta['vocab_size']} != model vocab {cfg.vocab_size}. "
                     "The SFT set and the base checkpoint use different tokenizers.")
if X.shape[1] > cfg.block_size:
    raise SystemExit(f"examples are {X.shape[1]} tokens but context is {cfg.block_size}")

rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(X))
n_val = max(1, int(len(X) * args.val_frac))
val_idx, train_idx = perm[:n_val], perm[n_val:]
steps_per_epoch = len(train_idx) // args.batch_size
max_steps = args.epochs * steps_per_epoch
print(f"{len(train_idx)} train / {len(val_idx)} val | {steps_per_epoch} steps/epoch "
      f"| {max_steps} total")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.95), weight_decay=0.0)


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = (step - args.warmup) / max(1, max_steps - args.warmup)
    return 0.1 * args.lr + 0.5 * (1 + math.cos(math.pi * prog)) * (0.9 * args.lr)


def batch(idx):
    x = torch.from_numpy(X[idx].astype(np.int64))
    y = torch.from_numpy(Y[idx].astype(np.int64))
    # the model grades logits[i] against targets[i], so shift by one to make it
    # predict the NEXT token rather than copy the current one
    return x[:, :-1].to(device), y[:, 1:].to(device)


@torch.no_grad()
def val_loss():
    model.eval()
    losses = []
    for s in range(0, len(val_idx), args.batch_size):
        chunk = val_idx[s:s + args.batch_size]
        if len(chunk) == 0:
            break
        x, y = batch(chunk)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(x, y)          # -100 positions ignored by cross_entropy
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save(path, epoch, val):
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "tokenizer": ck.get("tokenizer", "code_bpe_fim"),
                "tokenizer_path": ck.get("tokenizer_path", str(FIM_TOKENIZER)),
                "fim": ck.get("fim"), "base": args.base,
                "epoch": epoch, "val": val, "sft_data": args.data_dir}, path)


best = float("inf")
step = 0
t0 = time.time()
print(f"val before training: {val_loss():.4f}")
for epoch in range(1, args.epochs + 1):
    order = rng.permutation(train_idx)
    for s in range(steps_per_epoch):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch(order[s * args.batch_size:(s + 1) * args.batch_size])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        step += 1
        if step % args.log_interval == 0:
            dt = time.time() - t0
            print(f"epoch {epoch}/{args.epochs} | step {step}/{max_steps} "
                  f"| loss {loss.item():.3f} | lr {lr_at(step):.1e} "
                  f"| ETA {(max_steps - step) * dt / step / 60:.0f} min", flush=True)

    v = val_loss()
    save(args.out, epoch, v)
    print(f"--> epoch {epoch}: val {v:.4f} -> {args.out}", flush=True)
    if v < best:
        best = v
        save(args.out.replace(".pt", "_best.pt"), epoch, v)
        print(f"    new best -> {args.out.replace('.pt', '_best.pt')}", flush=True)

print(f"Done. best val {best:.4f}")
