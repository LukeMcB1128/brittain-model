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
"""
import os
import sys
import ast
import math
import argparse

import torch

from model import Brittain, GPTConfig
from tok_util import load_tokenizer

p = argparse.ArgumentParser()
p.add_argument("checkpoints", nargs="+")
p.add_argument("--code_text", default=None,
               help="raw code file for BPB (default: this repo's model.py)")
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


def load(path):
    ck = torch.load(path, map_location=device)
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
        logits, _ = model(x, y)
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
            out = model.generate(ids, max_new_tokens=80, temperature=0.2,
                                 top_p=0.95, repetition_penalty=1.0)
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


code_text = args.code_text or __file__.replace("eval_compare.py", "model.py")
code_sample = read_text(code_text, args.max_bytes)
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
