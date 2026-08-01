"""
Inference for BRITTAIN. Streams completions token-by-token.

    python3 sample.py                                  # newest checkpoint, interactive
    python3 sample.py brittain_235m_weights.pt         # pick a checkpoint
    python3 sample.py -p "def quicksort(arr):"         # one-shot
    python3 sample.py -f mymodule.py                   # continue a real file
    python3 sample.py -t 0.8                           # hotter

Defaults are temperature 0.4 / top_p 0.95 / repetition_penalty 1.12, which is what
eval_compare.py uses and what these models actually behave well at. The old
hardcoded 0.9/0.9/1.3 made them look far worse than they are — code has much lower
entropy than prose, so it wants a colder sampler.

Interactive mode takes MULTI-LINE input: paste or type as many lines as you like,
then a blank line to generate. Ctrl-C to quit.

No context migration hacks needed — RoPE means the model just works at its trained
context length (and degrades gracefully a bit beyond it).
"""
import os
import sys
import glob
import codecs
import argparse

import torch

from model import Brittain, GPTConfig
from tok_util import load_tokenizer


def newest_checkpoint():
    """Default to whatever was trained most recently, so `python3 sample.py` just works.

    Checkpoints live in weights/, but the training scripts write to the cwd, so
    both are searched — a run in progress is findable before it has been filed away.
    """
    found = glob.glob("weights/brittain_*.pt") + glob.glob("brittain_*.pt")
    found = sorted(found, key=os.path.getmtime, reverse=True)
    return found[0] if found else "weights/brittain_124m_best.pt"


ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", nargs="?", default=None)
ap.add_argument("-p", "--prompt", help="generate once from this text and exit")
ap.add_argument("-f", "--file", help="generate once continuing this file's contents")
ap.add_argument("-n", "--max_tokens", type=int, default=400)
ap.add_argument("-t", "--temperature", type=float, default=0.4)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--top_k", type=int, default=None)
ap.add_argument("-r", "--repetition_penalty", type=float, default=1.12)
ap.add_argument("--stop_blank", action="store_true",
                help="stop at the first blank line (how autocomplete behaves)")
args = ap.parse_args()

ckpt = args.checkpoint or newest_checkpoint()
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

ck = torch.load(ckpt, map_location=device)
cfg = GPTConfig(**ck["cfg"])
model = Brittain(cfg).to(device)
model.load_state_dict(ck["model"])
model.eval()
enc = load_tokenizer(ck)   # gpt2 for v1 ckpts, code BPE for v2

val = ck.get("val")
print(f"Loaded {ckpt} ({model.num_params():,} params) at iter {ck.get('iter', '?')}"
      + (f", val {val:.4f}" if isinstance(val, float) else ""))
print(f"{enc.name} vocab {enc.vocab_size} | ctx {cfg.block_size} | {device.type} | "
      f"temp {args.temperature} top_p {args.top_p} rep {args.repetition_penalty}")
print("-" * 70)


DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


def generate(prompt):
    """Stream a completion for one prompt. Keeps the last block_size tokens only.

    The prompt is echoed DIM and the model's own output bright, so it is always
    obvious which text the model actually produced. Only the part of the prompt
    that fits in the context is echoed — anything older was never seen.
    """
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    if ids.size(1) >= cfg.block_size:
        kept = cfg.block_size - 1
        print(f"[prompt is {ids.size(1)} tokens; the model sees only the last {kept}]")
        ids = ids[:, -kept:]
        prompt = enc.decode(ids[0].tolist())   # echo only what it actually sees
    print(DIM + prompt + RESET, end="", flush=True)
    # incremental UTF-8 decoder buffers multi-byte chars across tokens (no <?>)
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    emitted = ""
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        # stream() keeps the KV cache alive across tokens; looping over
        # generate(max_new_tokens=1) would rebuild it every token instead.
        for tok in model.stream(ids, args.max_tokens,
                                temperature=args.temperature, top_k=args.top_k,
                                top_p=args.top_p,
                                repetition_penalty=args.repetition_penalty):
            nxt = tok[0, -1].item()
            if nxt == enc.eot:                       # document boundary
                break
            piece = utf8.decode(enc.token_bytes(nxt))
            emitted += piece
            print(piece, end="", flush=True)
            # only after real output — models often open with a newline
            if args.stop_blank and emitted.strip() and "\n\n" in emitted:
                break
    print()
    if not emitted.strip():
        print("[no output — the model hit end-of-document immediately. This usually "
              "means the prompt is a COMPLETE file; truncate it mid-function to get "
              "a real completion.]")


if args.file:
    with open(args.file) as f:
        generate(f.read())
    sys.exit(0)

if args.prompt:
    generate(args.prompt)
    sys.exit(0)

print("Multi-line input — blank line to generate, Ctrl-C to quit.")
while True:
    try:
        lines = []
        while True:
            line = input("... " if lines else ">>> ")
            if not line:
                break
            lines.append(line)
        if not lines:
            continue
        print()
        generate("\n".join(lines) + "\n")
        print("-" * 70)
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
        break
