"""
Compare BRITTAIN checkpoints fairly, across different tokenizers.

    python3 eval_compare.py brittain_124m_best.pt brittain_235m_weights.pt ...

Three measurements:

1. BITS PER BYTE (the important one)
   You CANNOT compare raw val loss between these models. Cross-entropy is
   per-token, and the models use different tokenizers (50257 vs 32000) on
   different data. BRITTAIN-1's 3.25 and BRITTAIN-2's ~1.35 are not on the same
   scale at all.

   BPB normalises by raw UTF-8 bytes instead of tokens, which cancels the
   tokenizer out entirely:

       BPB = total_nll_nats / (ln(2) * n_bytes)

   Run every model over the SAME raw text and the numbers become comparable.
   Lower is better. This is what the literature reports for exactly this reason.

2. SYNTAX VALIDITY
   Generate completions from code prompts and check they parse with ast.parse().
   Directly measures "did it learn to write code". Nothing is executed — parsing
   is safe, running model-generated code would not be.

3. STATS
   Params, vocab, context, and tokenizer efficiency on the same sample.

Note: each model is evaluated at its own block_size, so a shorter-context model
eats more chunk boundaries. That's a real limitation of the model, not a bug in
the measurement, but it's worth knowing when reading the table.

THE CODE SAMPLE IS FROZEN. `data/eval_code.py` is a committed snapshot of model.py
taken 2026-07-31. It must never be edited. The default used to be the live
model.py, which meant BPB silently changed whenever the architecture was touched —
adding the KV cache moved the 235M's code BPB from 0.693 to 0.751 with no change to
the model at all. A benchmark that moves under you measures nothing. Prose uses
data/english.txt, which is already static.
"""
import os
import sys
import ast
import math
import warnings
import argparse

# Generated code frequently contains things like "\d" outside a raw string.
# ast.parse warns, we count it as valid Python (it is), and the warning is noise.
warnings.filterwarnings("ignore", category=SyntaxWarning)

import torch

from model import Brittain, GPTConfig
from tok_util import load_tokenizer
import model_bs

p = argparse.ArgumentParser()
p.add_argument("checkpoints", nargs="+")
p.add_argument("--code_text", default="data/eval_code.py",
               help="raw code file for BPB. Default is a FROZEN snapshot — see below")
p.add_argument("--prose_text", default="data/english.txt",
               help="raw English file for BPB")
p.add_argument("--max_bytes", type=int, default=200_000)
p.add_argument("--samples", type=int, default=20, help="completions per prompt")
args = p.parse_args()

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

CODE_PROMPTS = [
    "def fibonacci(n):\n",
    "def reverse_string(s):\n",
    "import os\n\ndef list_files(path):\n",
    "class Stack:\n    def __init__(self):\n",
    "def binary_search(arr, target):\n",
]


class _BSCfg:
    """Minimal cfg shim so BS models expose the same fields as GPTConfig."""
    def __init__(self, m):
        self.vocab_size, self.block_size = m.vocab, m.block


def load(path):
    ck = torch.load(path, map_location=device)
    # train_50m.bs checkpoints are a bare ModuleList state_dict (no 'cfg' key)
    # and use a different architecture — route them through model_bs.
    if not isinstance(ck, dict) or "cfg" not in ck:
        model, enc = model_bs.load(path, device)
        return model, _BSCfg(model), enc
    cfg = GPTConfig(**ck["cfg"])
    model = Brittain(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, load_tokenizer(ck)


def read_text(path, limit):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(limit)


@torch.no_grad()
def bits_per_byte(model, cfg, enc, text):
    """Total NLL over the text, normalised by its UTF-8 byte count."""
    n_bytes = len(text.encode("utf-8"))
    ids = enc.encode(text)
    if len(ids) < 2:
        return float("nan")
    block = cfg.block_size
    total_nll = 0.0
    # non-overlapping windows; every model is scored the same way
    for start in range(0, len(ids) - 1, block):
        chunk = ids[start:start + block + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        out = model(x, y) if isinstance(model, Brittain) else model(x)
        logits = out[0] if isinstance(out, tuple) else out
        # recompute unreduced so we can sum rather than average
        lp = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_nll += lp.item()
    return total_nll / (math.log(2) * n_bytes)


@torch.no_grad()
def syntax_validity(model, cfg, enc, n_samples):
    """Fraction of generated completions that parse as Python."""
    ok = 0
    total = 0
    for prompt in CODE_PROMPTS:
        ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        for _ in range(n_samples):
            # identical sampling for every model, or the comparison isn't fair
            if isinstance(model, Brittain):
                out = model.generate(ids, max_new_tokens=80, temperature=0.4,
                                     top_p=0.95, repetition_penalty=1.12)
            else:
                out = model.generate(ids, 80, temperature=0.4, top_p=0.95,
                                     repetition_penalty=1.12)
            gen = out[0, ids.size(1):].tolist()
            if enc.eot in gen:
                gen = gen[:gen.index(enc.eot)]
            text = prompt + enc.decode(gen)
            # trim to the last complete line — a cut-off final line is a
            # generation-length artifact, not a syntax failure
            text = text[:text.rfind("\n") + 1] if "\n" in text else text
            total += 1
            try:
                ast.parse(text)
                ok += 1
            except (SyntaxError, ValueError):
                pass
    return ok / max(1, total)


code_sample = read_text(args.code_text, args.max_bytes)
prose_sample = read_text(args.prose_text, args.max_bytes) \
    if os.path.exists(args.prose_text) else None

rows = []
for path in args.checkpoints:
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        continue
    print(f"evaluating {path} ...", flush=True)
    model, cfg, enc = load(path)
    row = {
        "name": os.path.basename(path),
        "params": model.num_params(),
        "vocab": cfg.vocab_size,
        "ctx": cfg.block_size,
        "tok": enc.name,
        "tok_eff": len(code_sample.encode()) / max(1, len(enc.encode(code_sample))),
        "bpb_code": bits_per_byte(model, cfg, enc, code_sample),
        "bpb_prose": bits_per_byte(model, cfg, enc, prose_sample) if prose_sample else float("nan"),
        "syntax": syntax_validity(model, cfg, enc, args.samples),
    }
    rows.append(row)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

print()
hdr = f"{'model':<28}{'params':>12}{'vocab':>8}{'ctx':>6}{'B/tok':>7}" \
      f"{'BPB code':>10}{'BPB prose':>11}{'syntax%':>9}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['name']:<28}{r['params']:>12,}{r['vocab']:>8}{r['ctx']:>6}"
          f"{r['tok_eff']:>7.2f}{r['bpb_code']:>10.3f}{r['bpb_prose']:>11.3f}"
          f"{100*r['syntax']:>8.0f}%")
print("\nBPB: lower is better, and IS comparable across tokenizers.")
print("Raw val loss is NOT comparable across these models — use BPB.")
