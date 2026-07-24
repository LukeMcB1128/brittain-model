"""
Build the BRITTAIN-2-coder pretraining corpus: The Stack (Python/JS/TS) mixed
with ~15% FineWeb-Edu English, tokenized with our 32k code BPE.

Why the English mix: a pure-code model gets strangely bad at natural language,
and we need it to read comments/docstrings and to follow instructions in the
later SFT stage. 15% keeps that ability without diluting the code.

Languages are round-robined (not Python-then-JS-then-TS) so the model sees a
mixed distribution throughout training rather than in blocks. The English ratio
is tracked in TOKENS and self-corrects batch to batch.

Tokenization uses tokenizers' encode_batch, which is parallel in Rust — no
multiprocessing, so none of the GIL-at-shutdown mess from prepare_fineweb.

Prereqs:
    hf auth login          # The Stack is a gated dataset
    python3 train_tokenizer.py     # writes data/code_bpe.json
Run:
    python3 prepare_code.py --tokens 2e8     # small TEST corpus first!
    python3 prepare_code.py --tokens 15e9    # the real one (~30GB, hours)
"""
import os
import sys
import argparse
import pickle

# xet backend has thrown SIGBUS on this box; use classic HTTP
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=float, default=15e9)
p.add_argument("--val_tokens", type=float, default=1e7)
p.add_argument("--english_frac", type=float, default=0.15)
p.add_argument("--dataset", type=str, default="bigcode/the-stack-dedup")
p.add_argument("--tokenizer", type=str, default="data/code_bpe.json")
p.add_argument("--langs", type=str, default="python,javascript,typescript",
               help="The Stack data_dir names; try Python,JavaScript,TypeScript if 404")
args = p.parse_args()

OUT = "./data"
os.makedirs(OUT, exist_ok=True)
LANGS = [l.strip() for l in args.langs.split(",")]
BATCH = 512                     # docs per encode_batch call

tok = Tokenizer.from_file(args.tokenizer)
EOT = tok.token_to_id("<|endoftext|>")
VOCAB = tok.get_vocab_size()
assert VOCAB < 65536, "vocab must fit in uint16"
print(f"tokenizer: {args.tokenizer} (vocab {VOCAB}, eot {EOT})")


def lang_stream(lang):
    ds = load_dataset(args.dataset, data_dir=f"data/{lang}", split="train", streaming=True)
    for ex in ds:
        text = ex.get("content") or ex.get("text") or ""
        if text.strip():
            yield text


def english_stream():
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    for ex in ds:
        if ex["text"].strip():
            yield ex["text"]


def build(val_path, train_path, val_target, train_target):
    """Stream once: first val_target tokens -> val (held out), the rest -> train."""
    code = [lang_stream(l) for l in LANGS]
    eng = english_stream()
    fval, ftrain = open(val_path, "wb"), open(train_path, "wb")
    n_val = n_train = 0
    n_code_tok = n_eng_tok = 0          # token counts drive the mix ratio
    rr = 0                              # round-robin index across languages
    exhausted = False

    while not exhausted and n_train < train_target:
        # --- fill a batch of documents, choosing source to hold the ratio ---
        buf, kinds = [], []
        while len(buf) < BATCH:
            total = n_code_tok + n_eng_tok
            want_eng = n_eng_tok < args.english_frac * max(1, total)
            try:
                if want_eng:
                    buf.append(next(eng)); kinds.append("eng")
                else:
                    buf.append(next(code[rr % len(code)])); kinds.append("code")
                    rr += 1
            except StopIteration:
                exhausted = True
                break
        if not buf:
            break

        # --- encode in parallel, then write ---
        for enc, kind in zip(tok.encode_batch(buf), kinds):
            ids = np.array(enc.ids + [EOT], dtype=np.uint16)
            if kind == "eng":
                n_eng_tok += len(ids)
            else:
                n_code_tok += len(ids)
            if n_val < val_target:
                fval.write(ids.tobytes()); n_val += len(ids)
            elif n_train < train_target:
                ftrain.write(ids.tobytes()); n_train += len(ids)
                if n_train % 200_000_000 < len(ids):
                    pct = 100 * n_eng_tok / max(1, n_code_tok + n_eng_tok)
                    print(f"  train {n_train/1e9:.2f}B / {train_target/1e9:.2f}B "
                          f"tokens ({pct:.0f}% english)", flush=True)
            else:
                break

    fval.close(); ftrain.close()
    return n_val, n_train, n_code_tok, n_eng_tok


if __name__ == "__main__":
    print(f"Building {args.tokens/1e9:.2f}B train + {args.val_tokens/1e6:.0f}M val ...")
    n_val, n_train, n_code, n_eng = build(
        os.path.join(OUT, "val.bin"), os.path.join(OUT, "train.bin"),
        args.val_tokens, args.tokens)
    with open(os.path.join(OUT, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": VOCAB, "tokenizer": "code_bpe"}, f)
    pct = 100 * n_eng / max(1, n_code + n_eng)
    print(f"Done. val {n_val/1e6:.0f}M | train {n_train/1e9:.2f}B tokens "
          f"({pct:.0f}% english). Wrote data/*.bin + meta.pkl")
    sys.stdout.flush()
    os._exit(0)
