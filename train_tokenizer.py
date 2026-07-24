"""
Train a 32k byte-level BPE tokenizer on CODE (for BRITTAIN-2-coder).

Why not reuse gpt2 BPE: it was trained on web prose, where long runs of spaces
basically never occur. Code is made of them. GPT-2 burns multiple tokens per
line on indentation and chops identifiers awkwardly. A BPE trained on code
learns single tokens for "\\n    ", "\\n        ", "def ", "function", "=>",
"self.", "const " etc. -> ~25% fewer tokens for the same code, which means
~25% more code per training step AND per context window, for free.

Smaller vocab (32k vs 50257) also frees ~14M params from the embedding table
to spend on actual transformer layers.

NOTE: we use a bare ByteLevel pre-tokenizer (no splitting regex) so BPE is free
to merge whitespace runs into single tokens. That's the whole trick for code.

Prereq (gated dataset):
    hf auth login
Run:
    python3 train_tokenizer.py                  # ~10-20 min on a few hundred MB
Writes data/code_bpe.json
"""
import os
import argparse

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

p = argparse.ArgumentParser()
p.add_argument("--vocab_size", type=int, default=32000)
p.add_argument("--sample_docs", type=int, default=200_000,
               help="how many code files to train the tokenizer on (it does NOT "
                    "need the whole corpus — a few hundred MB is plenty)")
p.add_argument("--english_docs", type=int, default=20_000,
               help="some English so prose/comments tokenize sanely too")
p.add_argument("--dataset", type=str, default="bigcode/the-stack-dedup")
p.add_argument("--out", type=str, default="data/code_bpe.json")
p.add_argument("--langs", type=str, default="python,javascript,typescript",
               help="The Stack data_dir names; try Python,JavaScript,TypeScript if 404")
args = p.parse_args()

os.makedirs("data", exist_ok=True)
LANGS = [l.strip() for l in args.langs.split(",")]
EOT = "<|endoftext|>"


def code_iter():
    """Stream a sample of each language, yielding file contents."""
    per_lang = max(1, args.sample_docs // len(LANGS))
    for lang in LANGS:
        ds = load_dataset(args.dataset, data_dir=f"data/{lang}",
                          split="train", streaming=True)
        n = 0
        for ex in ds:
            text = ex.get("content") or ex.get("text") or ""
            if text.strip():
                yield text
                n += 1
            if n >= per_lang:
                break
        print(f"  sampled {n} {lang} files", flush=True)


def english_iter():
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    n = 0
    for ex in ds:
        if ex["text"].strip():
            yield ex["text"]
            n += 1
        if n >= args.english_docs:
            break
    print(f"  sampled {n} English docs", flush=True)


def corpus():
    yield from code_iter()
    yield from english_iter()


tok = Tokenizer(models.BPE())
# Bare ByteLevel: no regex splitting, so runs of indentation CAN merge.
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size,
    special_tokens=[EOT],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes -> never OOV
    show_progress=True,
)

print(f"Training {args.vocab_size}-token BPE on code + English ...")
tok.train_from_iterator(corpus(), trainer=trainer)
tok.save(args.out)
print(f"Saved -> {args.out}  (vocab {tok.get_vocab_size()})")

# --- quick sanity check: how much better is this than gpt2 on real code? ---
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
    print(f"\nSample function -> gpt2: {gpt2} tokens | code BPE: {mine} tokens "
          f"({100*(gpt2-mine)/gpt2:.0f}% fewer)")
except Exception:
    print(f"\nSample function -> code BPE: {mine} tokens")
