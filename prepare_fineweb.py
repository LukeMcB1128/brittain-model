"""
Build a REAL pretraining corpus for the 124M cloud run: FineWeb-Edu (10B-token
sample) + your code folder mixed in. Run this ON THE CLOUD BOX, not the Mac
(10B tokens ~= 20 GB of .bin on disk).

FineWeb-Edu is high-quality, education-filtered web text — the standard corpus
people use to reproduce GPT-2 from scratch. gpt2 BPE tokenizer, same as the Mac
pipeline, so the model config is unchanged.

Setup on the box:
    pip install datasets tiktoken numpy tqdm
    python3 prepare_fineweb.py            # ~10B tokens (few hours of tokenizing)
    python3 prepare_fineweb.py --tokens 1e9   # smaller test corpus first

Writes data/train.bin, data/val.bin, data/meta.pkl.
"""
import os
import argparse
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=float, default=10e9, help="target token count")
parser.add_argument("--code_dir", type=str, default=os.path.expanduser("~/code"),
                    help="optional local code folder to mix in (skipped if missing)")
parser.add_argument("--val_tokens", type=float, default=5e6)
args = parser.parse_args()

OUT = "./data"
os.makedirs(OUT, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # document separator

def tokenize(doc):
    ids = enc.encode_ordinary(doc["text"])
    ids.append(EOT)
    return np.array(ids, dtype=np.uint16)

def write_stream(path, target_tokens):
    """Stream FineWeb-Edu, tokenize in parallel, append to one .bin until full."""
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    written = 0
    with open(path, "wb") as f, mp.Pool(max(1, os.cpu_count() - 2)) as pool:
        for arr in pool.imap(tokenize, ds, chunksize=16):
            f.write(arr.tobytes())
            written += len(arr)
            if written % 5_000_000 < len(arr):
                print(f"  {path}: {written/1e6:.1f}M tokens", flush=True)
            if written >= target_tokens:
                break
    return written

def append_code(path):
    """Optionally fold your code folder in (tokenized, appended)."""
    if not os.path.isdir(args.code_dir):
        print(f"  (no code dir at {args.code_dir}, skipping code mix)")
        return 0
    exts = {'.py', '.js', '.ts', '.tsx', '.java', '.swift', '.md'}
    written = 0
    with open(path, "ab") as f:
        for root, dirs, files in os.walk(args.code_dir):
            dirs[:] = [d for d in dirs if d not in
                       {'node_modules', '.git', '__pycache__', 'dist', 'build'}]
            for fn in files:
                if os.path.splitext(fn)[1].lower() not in exts:
                    continue
                try:
                    txt = open(os.path.join(root, fn), encoding="utf-8",
                               errors="ignore").read()
                except Exception:
                    continue
                if not txt.strip():
                    continue
                ids = np.array(enc.encode_ordinary(txt) + [EOT], dtype=np.uint16)
                f.write(ids.tobytes())
                written += len(ids)
    print(f"  appended {written/1e6:.1f}M code tokens")
    return written

if __name__ == "__main__":
    print(f"Building val ({args.val_tokens/1e6:.0f}M tokens)...")
    write_stream(os.path.join(OUT, "val.bin"), args.val_tokens)
    print(f"Building train ({args.tokens/1e9:.1f}B tokens)...")
    n = write_stream(os.path.join(OUT, "train.bin"), args.tokens)
    n += append_code(os.path.join(OUT, "train.bin"))
    import pickle
    with open(os.path.join(OUT, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": enc.n_vocab, "tokenizer": "gpt2"}, f)
    print(f"Done. train ~= {n/1e9:.2f}B tokens. Wrote data/*.bin + meta.pkl")
