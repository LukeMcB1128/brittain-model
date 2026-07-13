"""
Build the training dataset with BYTE-LEVEL BPE tokenization (tiktoken gpt2).

Why this replaces the old char-level tokenizer:
  * Char-level (vocab 187) makes every token a single character. A 256-token
    context was only ~40 words, and the model burned most of its capacity
    learning to spell. BPE packs ~4 characters per token, so the SAME context
    length now holds ~4x more actual text, and the model sees whole words /
    common code fragments as single units.

Output: data/train.bin, data/val.bin (uint16 token ids) and data/meta.pkl.
"""
import os
import pickle

import numpy as np
import tiktoken

TARGET_DIR = os.path.expanduser("~/Downloads/Coding")
OUTPUT_DIR = "./data"
ALLOWED_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.java', '.swift', '.txt', '.md', '.json'}
IGNORED_DIRS = {'node_modules', '.git', '__pycache__', 'dist', 'build', '.idea', '.vscode'}

os.makedirs(OUTPUT_DIR, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")

print(f"Scanning {TARGET_DIR} ...")
chunks = []
for root, dirs, files in os.walk(TARGET_DIR):
    dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
    for file in files:
        if os.path.splitext(file)[1].lower() in ALLOWED_EXTENSIONS:
            try:
                with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if content.strip():
                    chunks.append(content)
            except Exception:
                continue

print(f"Files collected: {len(chunks)}")
all_text = "\n\n".join(chunks)

# Encode once; gpt2 BPE handles arbitrary text/code and all whitespace.
ids = enc.encode_ordinary(all_text)
print(f"Total tokens: {len(ids):,}  (chars: {len(all_text):,}, ~{len(all_text)/max(len(ids),1):.2f} chars/token)")

n = len(ids)
train_ids = np.array(ids[:int(n * 0.9)], dtype=np.uint16)
val_ids = np.array(ids[int(n * 0.9):], dtype=np.uint16)
train_ids.tofile(os.path.join(OUTPUT_DIR, 'train.bin'))
val_ids.tofile(os.path.join(OUTPUT_DIR, 'val.bin'))

with open(os.path.join(OUTPUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': enc.n_vocab, 'tokenizer': 'gpt2'}, f)

print(f"Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}")
print("Wrote data/train.bin, data/val.bin, data/meta.pkl")
if len(ids) < 5_000_000:
    print("\n[!] Only", f"{len(ids):,}", "tokens. This is very little data — the model will")
    print("    overfit fast. See RECOMMENDATIONS.md 'Data' for how to grow the corpus.")
