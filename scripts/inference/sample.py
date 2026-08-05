"""
Inference for BRITTAIN. Streams completions token-by-token.

    python3 scripts/inference/sample.py
    python3 scripts/inference/sample.py checkpoints/brittain2_235m_weights.pt
    python3 scripts/inference/sample.py -p "def quicksort(arr):"
    python3 scripts/inference/sample.py -f mymodule.py
    python3 scripts/inference/sample.py checkpoints/brittain2_235m_fim.pt \
        -p "def total(xs):\n" --suffix "    return result\n"
    python3 scripts/inference/sample.py -t 0.8

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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from brittain.model import Brittain, GPTConfig
from brittain import model_bs
from brittain.paths import CHECKPOINT_DIR
from brittain.tokenizer import load_tokenizer


def newest_checkpoint():
    """Default to the most recently trained checkpoint.

    Filed checkpoints live in checkpoints/, while an active run may still be
    writing to the repository root, so both locations are searched.
    """
    found = glob.glob(str(CHECKPOINT_DIR / "brittain*.pt"))
    found += glob.glob(str(PROJECT_ROOT / "brittain*.pt"))
    found = [p for p in found if "backup" not in os.path.basename(p).lower()]
    found = sorted(found, key=os.path.getmtime, reverse=True)
    return found[0] if found else str(CHECKPOINT_DIR / "brittain_124m_best.pt")


ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", nargs="?", default=None)
ap.add_argument("-p", "--prompt", help="generate once from this text and exit")
ap.add_argument("-f", "--file", help="generate once continuing this file's contents")
fim_group = ap.add_mutually_exclusive_group()
fim_group.add_argument("--suffix", help="text after the cursor for a FIM completion")
fim_group.add_argument("--suffix_file", help="read the text after the cursor from this file")
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
if isinstance(ck, dict) and "cfg" in ck:
    cfg = GPTConfig(**ck["cfg"])
    model = Brittain(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    enc = load_tokenizer(ck)   # gpt2 for v1 ckpts, code BPE for v2
    block = cfg.block_size
    iteration = ck.get("iter", "?")
    val = ck.get("val")
else:
    # The two 50M models were trained in BrittainScript and save a bare
    # ModuleList state_dict rather than the wrapped 235M checkpoint format.
    del ck
    model, enc = model_bs.load(ckpt, device)
    block = model.block
    iteration = "BrittainScript checkpoint"
    val = None

print(f"Loaded {ckpt} ({model.num_params():,} params) at {iteration}"
      + (f", val {val:.4f}" if isinstance(val, float) else ""))
print(f"{enc.name} vocab {enc.vocab_size} | ctx {block} | {device.type} | "
      f"temp {args.temperature} top_p {args.top_p} rep {args.repetition_penalty}")
if enc.has_fim:
    print("FIM tokenizer detected | use --suffix or --suffix_file to provide right context")

# This script feeds the prompt to the model UNTOUCHED. An SFT checkpoint was
# trained to see the Alpaca template, so handing it a bare instruction makes it
# continue the sentence as prose instead of answering. It still produces roughly
# the right code, buried in an unterminated docstring and invented surrounding
# context — which reads as a broken model rather than a misused one.
#
# Same class of mistake in the other direction gave brittain_124m_sft "I am a
# person." for `def add(a, b):`. Mode is inferred the way serve.py infers it, so
# the two agree.
if any(tag in os.path.basename(ckpt).lower() for tag in ("sft", "instruct")):
    print("\n  !! This looks like an INSTRUCTION-TUNED checkpoint, and sample.py\n"
          "     sends prompts raw — no Alpaca template. Expect it to continue your\n"
          "     text rather than answer it. Use chat.py instead:\n"
          f"       python3 scripts/inference/chat.py {ckpt}\n")
print("-" * 70)


DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


def generate(prompt, suffix=None):
    """Stream a completion for one prompt. Keeps the last block_size tokens only.

    The prompt is echoed DIM and the model's own output bright, so it is always
    obvious which text the model actually produced. Only the part of the prompt
    that fits in the context is echoed — anything older was never seen.
    """
    if suffix is not None:
        if not enc.has_fim:
            raise SystemExit("--suffix requires a FIM checkpoint")
        prefix_ids = enc.encode(prompt)
        suffix_ids = enc.encode(suffix)
        room = block - len(suffix_ids) - 3
        if room < 1:
            raise SystemExit("the suffix alone is too long for this model's context")
        if len(prefix_ids) > room:
            print(f"[FIM prefix is {len(prefix_ids)} tokens; keeping the last {room}]")
            prefix_ids = prefix_ids[-room:]
            prompt = enc.decode(prefix_ids)
        packed = ([enc.fim_prefix] + prefix_ids + [enc.fim_suffix]
                  + suffix_ids + [enc.fim_middle])
        ids = torch.tensor([packed], dtype=torch.long, device=device)
    else:
        ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        if ids.size(1) >= block:
            kept = block - 1
            print(f"[prompt is {ids.size(1)} tokens; the model sees only the last {kept}]")
            ids = ids[:, -kept:]
            prompt = enc.decode(ids[0].tolist())   # echo only what it actually sees
    print(DIM + prompt + RESET, end="", flush=True)
    # incremental UTF-8 decoder buffers multi-byte chars across tokens (no <?>)
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    emitted = ""
    def token_stream():
        if isinstance(model, Brittain):
            # stream() keeps the KV cache alive across tokens.
            yield from model.stream(ids, args.max_tokens,
                                    temperature=args.temperature, top_k=args.top_k,
                                    top_p=args.top_p,
                                    repetition_penalty=args.repetition_penalty)
            return
        cur = ids
        for _ in range(args.max_tokens):
            cur = model.generate(cur[:, -block:], 1,
                                 temperature=args.temperature,
                                 top_p=args.top_p,
                                 repetition_penalty=args.repetition_penalty)
            yield cur[:, -1:]

    stop_ids = {enc.eot}
    if enc.has_fim:
        stop_ids.update((enc.fim_prefix, enc.fim_suffix, enc.fim_middle))
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for tok in token_stream():
            nxt = tok[0, -1].item()
            if nxt in stop_ids:                      # document/FIM boundary
                break
            piece = utf8.decode(enc.token_bytes(nxt))
            emitted += piece
            print(piece, end="", flush=True)
            # only after real output — models often open with a newline
            if args.stop_blank and emitted.strip() and "\n\n" in emitted:
                break
    if suffix is not None:
        print(DIM + suffix + RESET, end="")
    print()
    if not emitted.strip():
        print("[no output — the model hit end-of-document immediately. This usually "
              "means the prompt is a COMPLETE file; truncate it mid-function to get "
              "a real completion.]")


suffix = args.suffix
if args.suffix_file:
    with open(args.suffix_file) as f:
        suffix = f.read()

if args.file:
    with open(args.file) as f:
        generate(f.read(), suffix)
    sys.exit(0)

if args.prompt:
    generate(args.prompt, suffix)
    sys.exit(0)

if suffix is not None:
    ap.error("--suffix and --suffix_file require --prompt or --file")

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
