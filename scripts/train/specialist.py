"""
Specialize a trained model on BrittainScript.

    python3 scripts/train/specialist.py --base checkpoints/brittain_50m_bs.pt \
        --data_dir data/processed/bs_native --out checkpoints/xs_bs_native.pt
    python3 scripts/train/specialist.py --base checkpoints/brittain_50m_bs.pt \
        --data_dir data/processed/bs_mixed --out checkpoints/xs_bs_mixed.pt

Then judge them with scripts/evaluate/brittain_script.py, not only validation loss.

WHY THIS IS NOT train.py
train.py pretrains from scratch: random init, 6e-4, a cosine schedule sized to
thousands of iterations. Its resume path loads the model AND the optimizer state
AND the iteration counter, so aiming it at a finished checkpoint restarts partway
through a schedule that has already decayed to min_lr — and if that checkpoint's
iter exceeds max_iters the loop does not run at all. This starts from a base
checkpoint with a fresh counter and a fresh optimizer, which is what specializing
means. train_sft.py already does that; it just reads example-batched SFT data
rather than the packed .bin files prepare_bs_train.py writes.

ITERATIONS COME FROM EPOCHS, DELIBERATELY
The corpus is ~1.7M tokens. At train.py's mac_test defaults (12 x 8 x 1024 =
98,304 tokens per iteration) that is 18 iterations per epoch, so its max_iters of
2000 would be 110 passes over 19,700 programs — the model would memorise the
corpus outright and score wonderfully on any metric that does not check novelty.
Here you say --epochs and the iteration count is derived, with grad_accum
defaulting to 1 so a few epochs is still a few hundred optimizer steps rather
than a few dozen.

Epochs are notional: get_batch samples random windows from the packed data, so an
"epoch" means tokens_seen == dataset_tokens, not a shuffled pass over every row.

THE ARCHITECTURE COMES FROM THE BASE CHECKPOINT, NOT FROM A PRESET
Loading base weights into a differently shaped model is not possible, so the
config is read from the checkpoint and the presets are ignored entirely. The
tokenizer is likewise inherited: the corpus was tokenized with the same code BPE,
and a specialization cannot change vocab without discarding the embeddings it is
supposed to be building on.
"""
import os
import math
import time
import pickle
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from brittain.model import Brittain, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--base", required=True, help="checkpoint to specialize FROM")
p.add_argument("--data_dir", required=True, help="dir with train.bin/val.bin/meta.pkl")
p.add_argument("--out", required=True, help="checkpoint to write")
p.add_argument("--epochs", type=float, default=3.0)
p.add_argument("--lr", type=float, default=3e-5,
               help="low on purpose: nudge the model, do not overwrite it")
p.add_argument("--min_lr", type=float, default=None, help="default: lr/10")
p.add_argument("--batch_size", type=int, default=12)
p.add_argument("--grad_accum", type=int, default=1)
p.add_argument("--dropout", type=float, default=None,
               help="default: whatever the base was trained with")
p.add_argument("--warmup_frac", type=float, default=0.05)
p.add_argument("--eval_interval", type=int, default=50)
p.add_argument("--eval_iters", type=int, default=40)
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--grad_clip", type=float, default=1.0)
p.add_argument("--log_interval", type=int, default=25)
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--compile", action="store_true", help="torch.compile (CUDA only)")
args = p.parse_args()
Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)

MIN_LR = args.min_lr if args.min_lr is not None else args.lr / 10

torch.manual_seed(args.seed)
device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

ck = torch.load(args.base, map_location=device, weights_only=False)
cfg_dict = dict(ck["cfg"])
if args.dropout is not None:
    cfg_dict["dropout"] = args.dropout
cfg = GPTConfig(**cfg_dict)

if cfg.vocab_size != meta["vocab_size"]:
    raise SystemExit(
        f"vocab mismatch: base checkpoint {cfg.vocab_size}, data "
        f"{meta['vocab_size']}. A specialization inherits its tokenizer — "
        f"rebuild the .bin files with the tokenizer the base was trained on.")

block_size = cfg.block_size
tokens_per_iter = args.batch_size * args.grad_accum * block_size
max_iters = max(1, int(args.epochs * len(train_data) / tokens_per_iter))
warmup_iters = max(1, min(int(args.warmup_frac * max_iters), max_iters // 4))

model = Brittain(cfg).to(device)
model.load_state_dict(ck["model"])
raw_model = model
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.95), weight_decay=args.weight_decay)

print(f"--- specializing on {device.type} ---")
print(f"base      {args.base} ({model.num_params():,} params, "
      f"iter {ck.get('iter', '?')})")
print(f"data      {args.data_dir}: {len(train_data):,} train / "
      f"{len(val_data):,} val tokens")
print(f"schedule  {args.epochs} epochs = {max_iters} iters x {tokens_per_iter:,} "
      f"tokens (warmup {warmup_iters})")
print(f"lr        {args.lr:.1e} -> {MIN_LR:.1e}")
if max_iters < 50:
    print("WARNING: under 50 optimizer steps. Lower --grad_accum or --batch_size,")
    print("or the schedule barely runs before it is over.")

if args.compile and device.type == "cuda":
    try:
        model = torch.compile(model)
    except Exception as exc:
        print(f"torch.compile failed ({exc}); continuing uncompiled.")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (args.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_at(it):
    if it < warmup_iters:
        return args.lr * (it + 1) / warmup_iters
    if it > max_iters:
        return MIN_LR
    ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * ratio)) * (args.lr - MIN_LR)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(split)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_ckpt(path, it, val, best_val):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"iter": it, "model": raw_model.state_dict(),
                "optim": optimizer.state_dict(), "cfg": cfg.__dict__,
                "tokenizer": meta.get("tokenizer", "code_bpe"),
                "base": args.base, "data_dir": args.data_dir,
                "best_val": best_val, "val": val}, path)


best_val = float("inf")
stats = estimate_loss()
print(f"iter     0 | train {stats['train']:.4f} | val {stats['val']:.4f} "
      f"| (base model, before any update)", flush=True)

t0 = t_log = time.time()
for it in range(1, max_iters + 1):
    for g in optimizer.param_groups:
        g["lr"] = lr_at(it)

    optimizer.zero_grad(set_to_none=True)
    for _ in range(args.grad_accum):
        x, y = get_batch("train")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(x, y)
            loss = loss / args.grad_accum
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    if it % args.log_interval == 0:
        now = time.time()
        tps = tokens_per_iter * args.log_interval / (now - t_log)
        eta = (max_iters - it) * (now - t_log) / args.log_interval / 60
        print(f"iter {it:5d} | loss {loss.item() * args.grad_accum:.3f} "
              f"| {tps/1e3:.0f}k tok/s | ETA {eta:.1f} min", flush=True)
        t_log = now

    if it % args.eval_interval == 0 or it == max_iters:
        stats = estimate_loss()
        print(f"iter {it:5d} | train {stats['train']:.4f} | val {stats['val']:.4f} "
              f"| lr {lr_at(it):.2e} | {time.time()-t0:.0f}s", flush=True)
        save_ckpt(args.out, it, stats["val"], best_val)
        if stats["val"] < best_val:
            best_val = stats["val"]
            save_ckpt(args.out.replace(".pt", "_best.pt"), it, stats["val"], best_val)
        t_log = time.time()

print(f"\nDone. best val {best_val:.4f} -> {args.out}")
print("Val loss is the weaker signal. Score it by execution:")
print(f"    python3 scripts/evaluate/brittain_script.py "
      f"{args.out.replace('.pt', '_best.pt')} --n 200 --seed 1337")
print("and compare variants on RUNS AND IS NOVEL, not on the loss above.")
