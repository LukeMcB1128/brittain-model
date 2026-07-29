"""
Continued pretraining of the coder on fill-in-the-middle data.

    python3 prepare_fim.py --tokens 3e9
    python3 train_fim.py --base brittain_235m_best.pt --iters 5700

TWO THINGS THIS DOES THAT PLAIN RESUMING WOULD GET WRONG:

1. RESIZES THE VOCABULARY. prepare_fim.py appends three sentinel tokens, so the
   embedding must grow (32000 -> 32003). The trained rows are copied and the new
   ones initialised small. The embedding is TIED to the output head in model.py,
   so both are handled together — resize one and you resize both.

2. USES A FRESH LR SCHEDULE. The base run annealed its cosine down to min_lr at
   its final iteration. Resuming into that same schedule would jump the learning
   rate back up and undo the end-of-run gains. This starts a new, much smaller
   cosine (peak 1e-4 vs the 6e-4 of pretraining) — big enough to learn the FIM
   format, small enough not to wash out what the model already knows.

The base checkpoint is never modified; output goes to a new file.
"""
import os
import time
import math
import pickle
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from model import Brittain, GPTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="brittain_235m_best.pt")
ap.add_argument("--out", default="brittain_235m_fim.pt")
ap.add_argument("--data_dir", default="./data")
ap.add_argument("--iters", type=int, default=5700,
                help="5700 x 524288 tok ~= 3B tokens")
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--grad_accum", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4, help="peak LR (low: this is a nudge)")
ap.add_argument("--min_lr", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--eval_interval", type=int, default=200)
ap.add_argument("--eval_iters", type=int, default=50)
ap.add_argument("--log_interval", type=int, default=50)
ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--weight_decay", type=float, default=0.1)
ap.add_argument("--compile", action="store_true", default=True)
args = ap.parse_args()

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"--- FIM continued pretraining on {device} ---")

with open(os.path.join(args.data_dir, "fim_meta.pkl"), "rb") as f:
    meta = pickle.load(f)
new_vocab = meta["vocab_size"]
print(f"FIM data: vocab {new_vocab} (base {meta['base_vocab']}), "
      f"sentinels pre={meta['fim_prefix']} suf={meta['fim_suffix']} mid={meta['fim_middle']}")

train_data = np.memmap(os.path.join(args.data_dir, "fim_train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(args.data_dir, "fim_val.bin"), dtype=np.uint16, mode="r")

# ---------------- load base and grow the vocabulary ----------------
ck = torch.load(args.base, map_location="cpu")
old_cfg = dict(ck["cfg"])
old_vocab = old_cfg["vocab_size"]
block_size = old_cfg["block_size"]
if new_vocab < old_vocab:
    raise SystemExit(f"FIM vocab {new_vocab} smaller than base {old_vocab}")

cfg = GPTConfig(**{**old_cfg, "vocab_size": new_vocab})
model = Brittain(cfg)

sd = ck["model"]
# tok_emb and lm_head are the same tensor (tied). Copy trained rows into the new,
# larger table; rows for the sentinels keep their fresh small init.
emb_key = "tok_emb.weight" if "tok_emb.weight" in sd else "lm_head.weight"
old_emb = sd[emb_key]
if old_emb.shape[0] != old_vocab:
    raise SystemExit(f"embedding rows {old_emb.shape[0]} != cfg vocab {old_vocab}")
with torch.no_grad():
    model.tok_emb.weight[:old_vocab].copy_(old_emb)
grown = model.tok_emb.weight.detach().clone()
sd["tok_emb.weight"] = grown
sd["lm_head.weight"] = grown
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing or unexpected:
    print(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
model = model.to(device)
raw_model = model
print(f"Loaded {args.base}: {model.num_params():,} params, "
      f"vocab {old_vocab} -> {new_vocab} (+{new_vocab - old_vocab} rows)")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.95), weight_decay=args.weight_decay)
# deliberately NOT restoring ck['optim'] — momentum from the base run belongs to
# the old objective and the old (smaller) parameter shapes.

if args.compile and device.type == "cuda":
    try:
        print("Compiling ...")
        model = torch.compile(model)
    except Exception as exc:
        print(f"torch.compile failed ({exc}); continuing eager.")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (args.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_at(it):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    if it > args.iters:
        return args.min_lr
    r = (it - args.warmup) / max(1, args.iters - args.warmup)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - args.min_lr)


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


def save(path, it, val, best):
    torch.save({"iter": it, "model": raw_model.state_dict(),
                "optim": optimizer.state_dict(), "cfg": cfg.__dict__,
                "tokenizer": "code_bpe_fim",
                "tokenizer_path": meta.get("tokenizer_path", "data/code_bpe_fim.json"),
                "fim": {"prefix": meta["fim_prefix"], "suffix": meta["fim_suffix"],
                        "middle": meta["fim_middle"]},
                "val": val, "best_val": best, "base": args.base}, path)


tokens_per_iter = args.batch_size * args.grad_accum * block_size
print(f"{args.iters} iters x {tokens_per_iter:,} tok = "
      f"{args.iters * tokens_per_iter / 1e9:.2f}B tokens | peak lr {args.lr:g}")
best_val = float("inf")
t0 = t_log = time.time()
model.train()
for it in range(args.iters + 1):
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

    if it % args.log_interval == 0 and it > 0:
        nowt = time.time()
        tps = tokens_per_iter * args.log_interval / (nowt - t_log)
        eta = (args.iters - it) * (nowt - t_log) / args.log_interval / 60
        print(f"iter {it:5d} | loss {loss.item()*args.grad_accum:.3f} "
              f"| {tps/1e3:.0f}k tok/s | ETA {eta:.0f} min", flush=True)
        t_log = nowt

    if it % args.eval_interval == 0:
        s = estimate_loss()
        print(f"iter {it:5d} | train {s['train']:.4f} | val {s['val']:.4f} "
              f"| lr {lr_at(it):.2e} | {time.time()-t0:.0f}s", flush=True)
        save(args.out, it, s["val"], best_val)
        if s["val"] < best_val:
            best_val = s["val"]
            save(args.out.replace(".pt", "_best.pt"), it, s["val"], best_val)

print(f"Done. best val {best_val:.4f} -> {args.out}")
