"""
Compare BRITTAIN checkpoints fairly, across different tokenizers.

    python3 scripts/evaluate/compare.py checkpoints/brittain_124m_best.pt \
        checkpoints/brittain_235m_weights.pt

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

THE CODE SAMPLE IS FROZEN. `benchmarks/prompts/code.py` is a committed snapshot
of `src/brittain/model.py`
taken 2026-07-31. It must never be edited. The default used to be the live
model.py, which meant BPB silently changed whenever the architecture was touched —
adding the KV cache moved the 235M's code BPB from 0.693 to 0.751 with no change to
the model at all. A benchmark that moves under you measures nothing. Prose uses
`benchmarks/prompts/prose.txt`, which is already static.
"""
import os
import sys
import ast
import math
import warnings
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Generated code frequently contains things like "\d" outside a raw string.
# ast.parse warns, we count it as valid Python (it is), and the warning is noise.
warnings.filterwarnings("ignore", category=SyntaxWarning)

import torch

from brittain.loading import document_prefix, generate, load_any, resolve_device, strip_specials
from brittain.metrics import repetition_collapse
from brittain.model import Brittain
from brittain.model_v3 import Brittain3
from brittain.paths import BENCHMARK_PROMPTS_DIR

p = argparse.ArgumentParser()
p.add_argument("checkpoints", nargs="+")
p.add_argument("--code_text", default=str(BENCHMARK_PROMPTS_DIR / "code.py"),
               help="raw code file for BPB. Default is a FROZEN snapshot — see below")
p.add_argument("--prose_text", default=str(BENCHMARK_PROMPTS_DIR / "prose.txt"),
               help="raw English file for BPB")
p.add_argument("--max_bytes", type=int, default=200_000)
p.add_argument("--samples", type=int, default=100,
               help="completions per prompt. Syntax validity is a binomial "
                    "proportion: at 5 prompts x 20 samples sigma is ~5 points, so "
                    "a 7-point difference is noise. 100 puts sigma near 2.2.")
p.add_argument("--device", default=None)
p.add_argument("--block", type=int, default=None,
               help="score every model with this BPB window instead of its own "
                    "context. Use it when contexts differ: a longer window eats "
                    "fewer chunk boundaries, so an untrained 16K-capable model "
                    "would beat a 512-context one on window size alone.")
args = p.parse_args()

device = resolve_device(args.device)

CODE_PROMPTS = [
    "def fibonacci(n):\n",
    "def reverse_string(s):\n",
    "import os\n\ndef list_files(path):\n",
    "class Stack:\n    def __init__(self):\n",
    "def binary_search(arr, target):\n",
]


def read_text(path, limit):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(limit)


@torch.no_grad()
def bits_per_byte(model, block, enc, text):
    """Total NLL over the text, normalised by its UTF-8 byte count."""
    n_bytes = len(text.encode("utf-8"))
    ids = enc.encode(text)
    if len(ids) < 2:
        return float("nan")
    total_nll = 0.0
    # non-overlapping windows; every model is scored the same way
    for start in range(0, len(ids) - 1, block):
        chunk = ids[start:start + block + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        out = model(x, y) if isinstance(model, (Brittain, Brittain3)) else model(x)
        logits = out[0] if isinstance(out, tuple) else out
        # recompute unreduced so we can sum rather than average
        lp = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_nll += lp.item()
    return total_nll / (math.log(2) * n_bytes)


@torch.no_grad()
def syntax_validity(model, enc, n_samples):
    """Fraction of generated completions that parse, and how many collapsed.

    Returns (syntax_fraction, collapse_fraction). Collapse is measured on the
    same generations because a degenerate loop often still parses — the two
    numbers only mean something read together.
    """
    ok = 0
    collapsed = 0
    total = 0
    prefix = document_prefix(enc)
    for prompt in CODE_PROMPTS:
        ids = torch.tensor([enc.encode(prefix + prompt)], dtype=torch.long, device=device)
        for _ in range(n_samples):
            # identical sampling for every model, or the comparison isn't fair
            out = generate(model, ids, 80, temperature=0.4, top_p=0.95,
                           repetition_penalty=1.12)
            gen = strip_specials(enc, out[0, ids.size(1):].tolist())
            total += 1
            if repetition_collapse(gen):
                collapsed += 1
            text = prompt + enc.decode(gen)
            # trim to the last complete line — a cut-off final line is a
            # generation-length artifact, not a syntax failure
            text = text[:text.rfind("\n") + 1] if "\n" in text else text
            try:
                ast.parse(text)
                ok += 1
            except (SyntaxError, ValueError):
                pass
    return ok / max(1, total), collapsed / max(1, total)


code_sample = read_text(args.code_text, args.max_bytes)
prose_sample = read_text(args.prose_text, args.max_bytes) \
    if os.path.exists(args.prose_text) else None

rows = []
for path in args.checkpoints:
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        continue
    print(f"evaluating {path} ...", flush=True)
    model, block, enc = load_any(path, device)
    window = min(block, args.block) if args.block else block
    syntax, collapse = syntax_validity(model, enc, args.samples)
    row = {
        "name": os.path.basename(path),
        "params": model.num_params(),
        "vocab": enc.vocab_size,
        "ctx": block,
        "window": window,
        "tok": enc.name,
        "tok_eff": len(code_sample.encode()) / max(1, len(enc.encode(code_sample))),
        "bpb_code": bits_per_byte(model, window, enc, code_sample),
        "bpb_prose": bits_per_byte(model, window, enc, prose_sample) if prose_sample else float("nan"),
        "syntax": syntax,
        "collapse": collapse,
    }
    rows.append(row)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

print()
hdr = f"{'model':<28}{'params':>12}{'vocab':>8}{'ctx':>6}{'win':>7}{'B/tok':>7}" \
      f"{'BPB code':>10}{'BPB prose':>11}{'syntax%':>9}{'collapse%':>11}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['name']:<28}{r['params']:>12,}{r['vocab']:>8}{r['ctx']:>6}{r['window']:>7}"
          f"{r['tok_eff']:>7.2f}{r['bpb_code']:>10.3f}{r['bpb_prose']:>11.3f}"
          f"{100*r['syntax']:>8.0f}%{100*r['collapse']:>10.1f}%")
print("\nBPB: lower is better, and IS comparable across tokenizers.")
print("Raw val loss is NOT comparable across these models — use BPB.")
if len({r["window"] for r in rows}) > 1:
    print("\nWARNING: BPB windows differ across these models, so the BPB column is "
          "NOT a like-for-like comparison. Re-run with --block "
          f"{min(r['window'] for r in rows)} to equalise it.")
