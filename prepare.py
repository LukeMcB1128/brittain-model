"""
Build the training dataset: BYTE-LEVEL BPE (tiktoken gpt2), mixing English prose
with your code.

Two big changes from the original char-level prepare.py:
  1. BPE tokenizer (vocab 50257) instead of char-level (vocab 187): ~4 chars per
     token, so the same context holds ~4x more real text.
  2. English + code MIX. A code-only, ~0.7M-token corpus can't teach coherent
     language. We add ~9M tokens of public-domain English (run get_english.py
     first) and upweight the code to CODE_FRACTION of the final mix.

Junk filtering: lockfiles, minified bundles, and giant/minified JSON are skipped
(the original run was ~65% package-lock.json — pure noise the model memorized).

Run:  python3 get_english.py   (once, to fetch English)
      python3 prepare.py
"""
import os
import pickle
import random

import numpy as np
import tiktoken

TARGET_DIR = os.path.expanduser("~/Downloads/Coding")
OUTPUT_DIR = "./data"
ENGLISH_FILE = os.path.join(OUTPUT_DIR, "english.txt")
CODE_FRACTION = 0.25          # target share of the mix that is your code
MAX_CODE_REPEAT = 4           # cap on how many times code is duplicated
SEED = 1337

ALLOWED_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.java', '.swift', '.md'}
IGNORED_DIRS = {'node_modules', '.git', '__pycache__', 'dist', 'build', '.idea', '.vscode'}
# lockfiles / generated files that are noise, not code:
SKIP_NAMES = {'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'composer.lock',
              'cargo.lock', 'poetry.lock', 'gemfile.lock'}
MAX_FILE_BYTES = 1_000_000    # skip huge (usually generated) files
MAX_LINE_LEN = 5000           # a very long line => minified bundle

os.makedirs(OUTPUT_DIR, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
random.seed(SEED)


def looks_like_junk(name, content):
    low = name.lower()
    if low in SKIP_NAMES or low.endswith(('.min.js', '.min.css')):
        return True
    if len(content) > MAX_FILE_BYTES:
        return True
    # minified / single-line data blobs
    if content and max((len(l) for l in content.splitlines()), default=0) > MAX_LINE_LEN:
        return True
    return False


# ---- collect code documents ----
print(f"Scanning {TARGET_DIR} for code ...")
code_docs, skipped = [], 0
for root, dirs, files in os.walk(TARGET_DIR):
    dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
    for file in files:
        if os.path.splitext(file)[1].lower() not in ALLOWED_EXTENSIONS:
            continue
        try:
            with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            continue
        if not content.strip():
            continue
        if looks_like_junk(file, content):
            skipped += 1
            continue
        code_docs.append(content)
print(f"  kept {len(code_docs)} code files, skipped {skipped} (junk/minified/lockfiles)")

# ---- English documents (chunk the big file so shuffling can interleave) ----
english_docs = []
if os.path.exists(ENGLISH_FILE):
    with open(ENGLISH_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        eng = f.read()
    CHUNK = 8000
    english_docs = [eng[i:i+CHUNK] for i in range(0, len(eng), CHUNK)]
    print(f"  loaded English: {len(eng)/1e6:.1f} MB in {len(english_docs)} chunks")
else:
    print("  [!] data/english.txt not found — run get_english.py first for an "
          "English mix.\n      Proceeding with code only (will be incoherent).")

# ---- tokenize ----
def toks(docs):
    return sum((enc.encode_ordinary(d) for d in docs), [])

code_tokens = toks(code_docs)
english_tokens_docs = english_docs  # keep as docs for shuffling
n_english = sum(len(enc.encode_ordinary(d)) for d in english_docs) if english_docs else 0
n_code = len(code_tokens)
print(f"  code tokens: {n_code:,} | english tokens: {n_english:,}")

# ---- choose how many times to repeat code to hit CODE_FRACTION ----
repeat = 1
if n_code > 0 and n_english > 0:
    # solve n_code*r / (n_english + n_code*r) = CODE_FRACTION
    target = CODE_FRACTION * n_english / ((1 - CODE_FRACTION) * n_code)
    repeat = max(1, min(MAX_CODE_REPEAT, round(target)))
final_code_frac = (n_code * repeat) / max(1, n_code * repeat + n_english)
print(f"  repeating code x{repeat} -> code is {final_code_frac*100:.0f}% of the mix")

# ---- build shuffled document list, then a single token stream ----
docs = list(english_docs) + code_docs * repeat
random.shuffle(docs)
ids = []
for d in docs:
    ids.extend(enc.encode_ordinary(d))
print(f"  total tokens: {len(ids):,}")

n = len(ids)
train_ids = np.array(ids[:int(n * 0.9)], dtype=np.uint16)
val_ids = np.array(ids[int(n * 0.9):], dtype=np.uint16)
train_ids.tofile(os.path.join(OUTPUT_DIR, 'train.bin'))
val_ids.tofile(os.path.join(OUTPUT_DIR, 'val.bin'))
with open(os.path.join(OUTPUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': enc.n_vocab, 'tokenizer': 'gpt2'}, f)

print(f"\nTrain: {len(train_ids):,} tokens | Val: {len(val_ids):,} tokens")
print("Wrote data/train.bin, data/val.bin, data/meta.pkl")
