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

THE VOCAB GROWS. The three sentinels are appended to tokenizer.json, so
vocab 32000 -> 32003, written as tokenizer_fim.json. scripts/train/fim.py resizes the
model's (tied) embedding to match, copying the trained rows and initialising the
three new ones.

Prereqs:  hf auth login  (The Stack is gated)
Run on the box:
    python3 scripts/prepare/prepare_fim.py --tokens 2e8
    python3 scripts/prepare/prepare_fim.py --tokens 3e9
"""
import os
import sys
import random
import pickle
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# xet backend has thrown SIGBUS on this box; use classic HTTP
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, AddedToken
from brittain.paths import BASE_TOKENIZER, FIM_TOKENIZER, PROCESSED_DATA_DIR

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
p.add_argument("--tokenizer_in", type=str, default=str(BASE_TOKENIZER))
p.add_argument("--tokenizer_out", type=str, default=str(FIM_TOKENIZER))
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

OUT = str(PROCESSED_DATA_DIR / "fim")
os.makedirs(OUT, exist_ok=True)
LANGS = [l.strip() for l in args.langs.split(",")]
BATCH = 256
MAX_DOC_CHARS = 60_000
rng = random.Random(args.seed)

# A FIM sequence is only useful if the whole prefix/suffix/middle triple lands
# inside ONE training window — get_batch() takes a random block_size slice of the
# flat .bin, so a longer sequence is seen as fragments with orphaned sentinels,
# and the model learns the sentinels are noise rather than structure.
#
# The window is measured in TOKENS, not characters. Bytes-per-token varies hugely
# with content (~3.2 on average prose-ish code, but near 1.0 on dense punctuation
# and indentation), so a character cap that is right on average overshoots badly
# on the worst documents. Capping tokens bounds it directly.
#
# Budget against the 1024 context: 900 window + 4 sentinels, leaving ~120 for the
# re-tokenization overhead of splitting one span into three (each cut can land
# mid-token and cost a few extra).
#
# Long files still contribute — as random windows, not as one oversized sequence.
# Plain causal documents are deliberately NOT capped: crossing chunk boundaries is
# exactly what left-to-right training should learn from.
FIM_MAX_TOKENS = 900

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
    """Split at two character offsets and emit a PSM or SPM token sequence.

    Oversized documents are windowed in token space first (see FIM_MAX_TOKENS) so
    the emitted triple fits in one training context. The window is decoded back to
    text before splitting, keeping the three cuts at CHARACTER level — a cut must
    never land inside a token.
    """
    if len(text) < 40:
        return None
    ids = tok.encode(text, add_special_tokens=False).ids
    if len(ids) > FIM_MAX_TOKENS:
        start = rng.randrange(0, len(ids) - FIM_MAX_TOKENS + 1)
        text = tok.decode(ids[start:start + FIM_MAX_TOKENS])
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
    fim_max = 0          # longest FIM sequence emitted; must stay under block_size
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
                    fim_max = max(fim_max, len(arr))
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
    return n_val, n_train, n_code_tok, n_eng_tok, n_fim_tok, fim_max


if __name__ == "__main__":
    print(f"Building {args.tokens/1e9:.2f}B FIM train + {args.val_tokens/1e6:.0f}M val ...")
    n_val, n_train, n_code, n_eng, n_fim, fim_max = build(
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
    print(f"Wrote {OUT}/fim_train.bin, fim_val.bin, and fim_meta.pkl")

    # GATE: every FIM triple must fit in one training window, or the model sees
    # orphaned sentinels and learns nothing from them.
    print(f"longest FIM sequence: {fim_max} tokens (FIM_MAX_TOKENS={FIM_MAX_TOKENS})")
    if fim_max >= 1024:
        print(f"  WARNING: {fim_max} >= the 1024-token context. FIM triples will be "
              f"split across windows and the sentinels will not teach structure. "
              f"Lower FIM_MAX_TOKENS and rebuild before training.")
    else:
        print("  OK — every FIM triple fits inside a 1024-token context.")
    sys.stdout.flush()
    os._exit(0)
