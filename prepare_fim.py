"""
Build fill-in-the-middle training data for continued pretraining of the coder.

WHY: normal causal training only ever teaches `prefix -> continuation`. IDE
autocomplete usually needs `prefix + suffix -> middle`, because your cursor sits
in the middle of a file. A model with no FIM training literally cannot use the
code after the cursor. This produces data that teaches it to.

FORMAT: each converted document becomes

    PSM:  <fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>MIDDLE<eot>
    SPM:  <fim_prefix><fim_suffix>SUFFIX<fim_middle>PREFIX MIDDLE<eot>

Both orderings are used (StarCoder does the same) — SPM helps when the prefix is
long, since the model sees the suffix before committing to the middle. Splits are
made at CHARACTER level and each piece tokenized separately, so a split never
lands inside a token.

--fim_rate controls the mix; the rest of the documents stay ordinary causal text.
Keeping ~50% plain matters: a pure-FIM diet degrades ordinary left-to-right
completion, which is still what end-of-line autocomplete uses.

THE VOCAB GROWS. The three sentinels are appended to code_bpe.json, so
vocab 32000 -> 32003, written as data/code_bpe_fim.json. train_fim.py resizes the
model's (tied) embedding to match, copying the trained rows and initialising the
three new ones.

Prereqs:  hf auth login  (The Stack is gated)
Run on the box:
    python3 prepare_fim.py --tokens 2e8      # small test first
    python3 prepare_fim.py --tokens 3e9      # the real continued-pretraining set
"""
import os
import sys
import random
import pickle
import argparse

# xet backend has thrown SIGBUS on this box; use classic HTTP
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, AddedToken

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=float, default=3e9)
p.add_argument("--val_tokens", type=float, default=5e6)
p.add_argument("--fim_rate", type=float, default=0.5,
               help="fraction of documents converted to FIM")
p.add_argument("--spm_rate", type=float, default=0.5,
               help="of the FIM documents, fraction using SPM ordering")
p.add_argument("--english_frac", type=float, default=0.10)
p.add_argument("--dataset", type=str, default="bigcode/the-stack-dedup")
p.add_argument("--langs", type=str, default="python,javascript,typescript")
p.add_argument("--tokenizer_in", type=str, default="data/code_bpe.json")
p.add_argument("--tokenizer_out", type=str, default="data/code_bpe_fim.json")
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

OUT = "./data"
os.makedirs(OUT, exist_ok=True)
LANGS = [l.strip() for l in args.langs.split(",")]
BATCH = 256
MAX_DOC_CHARS = 60_000
rng = random.Random(args.seed)

FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<fim_prefix>", "<fim_suffix>", "<fim_middle>"

# ---- extend the tokenizer with the three sentinels ----
tok = Tokenizer.from_file(args.tokenizer_in)
base_vocab = tok.get_vocab_size()
added = tok.add_special_tokens([AddedToken(t, normalized=False, special=True)
                               for t in (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE)])
tok.save(args.tokenizer_out)
VOCAB = tok.get_vocab_size()
EOT = tok.token_to_id("<|endoftext|>")
PRE = tok.token_to_id(FIM_PREFIX)
SUF = tok.token_to_id(FIM_SUFFIX)
MID = tok.token_to_id(FIM_MIDDLE)
assert None not in (EOT, PRE, SUF, MID), "sentinel ids missing"
assert VOCAB < 65536, "vocab must fit in uint16"
print(f"tokenizer {args.tokenizer_in} vocab {base_vocab} -> {VOCAB} "
      f"(+{added} sentinels)  pre={PRE} suf={SUF} mid={MID}")
print(f"wrote {args.tokenizer_out}")


def lang_stream(lang):
    ds = load_dataset(args.dataset, data_dir=f"data/{lang}", split="train", streaming=True)
    for ex in ds:
        text = ex.get("content") or ex.get("text") or ""
        if text.strip():
            yield text[:MAX_DOC_CHARS]


def english_stream():
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    for ex in ds:
        if ex["text"].strip():
            yield ex["text"][:MAX_DOC_CHARS]


def to_fim(text):
    """Split at two character offsets and emit a PSM or SPM token sequence."""
    n = len(text)
    if n < 40:
        return None
    a, b = sorted(rng.sample(range(1, n), 2))
    prefix, middle, suffix = text[:a], text[a:b], text[b:]
    if not middle.strip():
        return None
    pids = tok.encode(prefix, add_special_tokens=False).ids
    mids = tok.encode(middle, add_special_tokens=False).ids
    sids = tok.encode(suffix, add_special_tokens=False).ids
    if rng.random() < args.spm_rate:
        # SPM: suffix first, then prefix+middle as the continuation
        return [PRE, SUF] + sids + [MID] + pids + mids + [EOT]
    return [PRE] + pids + [SUF] + sids + [MID] + mids + [EOT]


def encode_doc(text, is_code):
    """FIM-transform code documents at fim_rate; everything else stays causal."""
    if is_code and rng.random() < args.fim_rate:
        ids = to_fim(text)
        if ids is not None:
            return ids, True
    return tok.encode(text, add_special_tokens=False).ids + [EOT], False


def build(val_path, train_path, val_target, train_target):
    code = [lang_stream(l) for l in LANGS]
    eng = english_stream()
    fval, ftrain = open(val_path, "wb"), open(train_path, "wb")
    n_val = n_train = 0
    n_code_tok = n_eng_tok = n_fim_tok = 0
    rr = 0
    exhausted = False

    while not exhausted and n_train < train_target:
        buf = []
        while len(buf) < BATCH:
            total = n_code_tok + n_eng_tok
            want_eng = n_eng_tok < args.english_frac * max(1, total)
            try:
                if want_eng:
                    buf.append((next(eng), False))
                else:
                    buf.append((next(code[rr % len(code)]), True))
                    rr += 1
            except StopIteration:
                exhausted = True
                break
        if not buf:
            break

        for text, is_code in buf:
            ids, was_fim = encode_doc(text, is_code)
            arr = np.array(ids, dtype=np.uint16)
            if is_code:
                n_code_tok += len(arr)
                if was_fim:
                    n_fim_tok += len(arr)
            else:
                n_eng_tok += len(arr)
            if n_val < val_target:
                fval.write(arr.tobytes()); n_val += len(arr)
            elif n_train < train_target:
                ftrain.write(arr.tobytes()); n_train += len(arr)
                if n_train % 100_000_000 < len(arr):
                    fp = 100 * n_fim_tok / max(1, n_code_tok)
                    ep = 100 * n_eng_tok / max(1, n_code_tok + n_eng_tok)
                    print(f"  train {n_train/1e9:.2f}B / {train_target/1e9:.2f}B "
                          f"({fp:.0f}% of code is FIM, {ep:.0f}% english)", flush=True)
            else:
                break

    fval.close(); ftrain.close()
    return n_val, n_train, n_code_tok, n_eng_tok, n_fim_tok


if __name__ == "__main__":
    print(f"Building {args.tokens/1e9:.2f}B FIM train + {args.val_tokens/1e6:.0f}M val ...")
    n_val, n_train, n_code, n_eng, n_fim = build(
        os.path.join(OUT, "fim_val.bin"), os.path.join(OUT, "fim_train.bin"),
        args.val_tokens, args.tokens)
    with open(os.path.join(OUT, "fim_meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": VOCAB, "tokenizer": "code_bpe_fim",
                     "tokenizer_path": args.tokenizer_out,
                     "fim_prefix": PRE, "fim_suffix": SUF, "fim_middle": MID,
                     "base_vocab": base_vocab}, f)
    print(f"Done. val {n_val/1e6:.0f}M | train {n_train/1e9:.2f}B tokens "
          f"({100*n_fim/max(1,n_code):.0f}% of code tokens are FIM, "
          f"{100*n_eng/max(1,n_code+n_eng):.0f}% english)")
    print("Wrote data/fim_train.bin, data/fim_val.bin, data/fim_meta.pkl")
    sys.stdout.flush()
    os._exit(0)
