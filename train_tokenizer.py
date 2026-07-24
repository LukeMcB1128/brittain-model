"""
Train a 32k byte-level BPE tokenizer on CODE (for BRITTAIN-2-coder).

Why not reuse gpt2 BPE: GPT-2's *merges* were learned on prose, where long runs
of indentation basically never occur, so it has no token for "\\n        " and
burns several tokens per indented line. Retraining the merges on code gives
single tokens for indentation runs, "def ", "function", "=>", "self." etc ->
~25% fewer tokens for the same code. That's ~25% more code per training step
AND per context window, for free. A 32k vocab (vs 50257) also frees ~14M params
from the embedding table to spend on transformer layers.

TWO PHASES, deliberately separate so a network hiccup can't kill training:
  1. sample  -> streams The Stack to a local JSONL file (constant memory, and
                skipped on re-run, so retries are free)
  2. train   -> trains BPE from that local file

JSONL (not plain text) because the trainer must see WHOLE FILES as single
sequences — a line-based format would put a boundary at every newline and make
"\\n    " unlearnable, defeating the entire point.

Prereq (gated dataset):
    hf auth login
Run:
    python3 train_tokenizer.py --langs python,javascript,typescript
Writes data/code_bpe.json (and data/tok_sample.jsonl)
"""
import os
import json
import argparse

# The xet backend memory-maps download chunks and has thrown SIGBUS here; the
# classic HTTP path is slower but reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

p = argparse.ArgumentParser()
p.add_argument("--vocab_size", type=int, default=32000)
p.add_argument("--code_mb", type=float, default=300, help="MB of code to sample")
p.add_argument("--english_mb", type=float, default=50, help="MB of English to sample")
p.add_argument("--dataset", type=str, default="bigcode/the-stack-dedup")
p.add_argument("--langs", type=str, default="python,javascript,typescript")
p.add_argument("--sample", type=str, default="data/tok_sample.jsonl")
p.add_argument("--out", type=str, default="data/code_bpe.json")
args = p.parse_args()

os.makedirs("data", exist_ok=True)
LANGS = [l.strip() for l in args.langs.split(",")]
EOT = "<|endoftext|>"
MAX_DOC_CHARS = 50_000          # cap giant/minified files


def write_stream(f, it, budget_bytes, label):
    wrote = 0
    for text in it:
        text = (text or "")[:MAX_DOC_CHARS]
        if not text.strip():
            continue
        f.write(json.dumps({"t": text}) + "\n")
        wrote += len(text)
        if wrote >= budget_bytes:
            break
    print(f"  {label}: {wrote/1e6:.0f} MB", flush=True)


def phase1_sample():
    if os.path.exists(args.sample) and os.path.getsize(args.sample) > 1_000_000:
        print(f"Reusing existing sample {args.sample} "
              f"({os.path.getsize(args.sample)/1e6:.0f} MB) — delete it to resample.")
        return
    print("Phase 1: sampling to", args.sample, flush=True)
    per_lang = args.code_mb * 1e6 / len(LANGS)
    with open(args.sample, "w") as f:
        for lang in LANGS:
            ds = load_dataset(args.dataset, data_dir=f"data/{lang}",
                              split="train", streaming=True)
            write_stream(f, (ex.get("content") or ex.get("text") or "" for ex in ds),
                         per_lang, lang)
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        write_stream(f, (ex["text"] for ex in ds), args.english_mb * 1e6, "english")


def docs():
    with open(args.sample) as f:
        for line in f:
            yield json.loads(line)["t"]


phase1_sample()

print("Phase 2: training BPE ...", flush=True)
tok = Tokenizer(models.BPE())
# ByteLevel's regex already groups a RUN of whitespace into one piece; training
# the merges on code is what makes those runs single tokens.
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size,
    special_tokens=[EOT],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   # all 256 bytes -> never OOV
    show_progress=True,
)
tok.train_from_iterator(docs(), trainer=trainer)
tok.save(args.out)
print(f"Saved -> {args.out}  (vocab {tok.get_vocab_size()})")

# --- GATE 1: is this actually better than gpt2 on real code? ---
SAMPLE = '''def process(items):
    results = []
    for item in items:
        if item.valid:
            results.append(item.value)
    return results
'''
mine = len(tok.encode(SAMPLE).ids)
try:
    import tiktoken
    gpt2 = len(tiktoken.get_encoding("gpt2").encode_ordinary(SAMPLE))
    print(f"\nGATE 1 -> gpt2: {gpt2} tokens | code BPE: {mine} tokens "
          f"({100*(gpt2-mine)/gpt2:.0f}% fewer)")
except Exception:
    print(f"\ncode BPE: {mine} tokens")
