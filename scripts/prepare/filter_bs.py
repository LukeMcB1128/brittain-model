"""
Turn the raw bs corpus into the file you actually train on.

    python3 scripts/prepare/filter_bs.py
    python3 scripts/prepare/filter_bs.py --stats

WHY THIS IS A SEPARATE PASS, NOT A CHECK INSIDE prepare_bs.py
Collection is network-bound and takes hours; filtering 10M tokens takes seconds.
Baking these thresholds into the collector would mean re-streaming The Stack every
time one of them turned out to be wrong. So bs_corpus.jsonl stays the raw capture
— append-only, resumable, never filtered — and this produces the training file
from it. Change a threshold, re-run, get a new training set in seconds.

WHAT IT REMOVES, AND WHY IT IS 80% OF THE TOKENS
py2bs accepts what it can translate, and the two things a restrictive transpiler
always handles are trivial fragments and pure data literals. Everything with real
logic hits a rejected construct. So the corpus is not a sample of Python — it is a
sample of the part of Python that is barely code. Measured on 10.0M raw tokens:

    data blob              30.7%   base64 PNGs, lookup matrices, unicode tables
    no logic               25.5%   nothing but literal assignments
    giant string literal   25.0%
    too small               0.7%   __all__ = [], __version__ = "0.1.0"
    ------------------------------------------------------------------
    usable                 18.0%   1.80M tokens, 20,026 programs

The single largest row was a 19,710-token FLAVOR_MATRIX of 0s and 1s. That ratio
held to within 0.6 points across a doubling of the corpus, so treat 18% as the
pipeline's real yield: to train on N tokens, collect about 5.5N.

A model trained on the unfiltered file would spend most of its capacity learning
to continue base64.
"""
import os
import re
import json
import argparse
import collections
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.paths import BS_CORPUS_DIR

DEFAULT_IN = str(BS_CORPUS_DIR / "bs_corpus.jsonl")

p = argparse.ArgumentParser()
p.add_argument("--in", dest="src", default=DEFAULT_IN)
p.add_argument("--out", default=None, help="default: <in> with .clean.jsonl")
p.add_argument("--stats", action="store_true", help="report only, write nothing")
p.add_argument("--min_tokens", type=int, default=12,
               help="below this a row is a fragment, not a program")
p.add_argument("--max_data_frac", type=float, default=0.5,
               help="max share of characters inside [] or {} — catches data tables")
p.add_argument("--max_string_len", type=int, default=400,
               help="reject rows holding a string literal this long (base64, blobs)")
p.add_argument("--val_frac", type=float, default=0.05,
               help="held-out share; 0 writes a single undivided file")
args = p.parse_args()

OUT = args.out or args.src.replace(".jsonl", "") + ".clean.jsonl"
OUT_TRAIN = OUT.replace(".jsonl", "") + ".train.jsonl"
OUT_VAL = OUT.replace(".jsonl", "") + ".val.jsonl"


def is_val(row):
    """Split on the CONTENT HASH, not a shuffle or a line number.

    The corpus is appended to across resumed runs, so a positional or random
    split silently reshuffles every time the file grows — and a program that was
    held out yesterday lands in training today, which quietly invalidates every
    number measured against it. Hashing the content pins each program to one side
    for good, no matter how the file is rebuilt or reordered.
    """
    return int(row["hash"], 16) % 10_000 < args.val_frac * 10_000

# A row must contain at least one of these to count as a program rather than a
# pile of assignments. push() is in the list because a script whose only action is
# printing is still a program; `x = 1` repeated 400 times is not.
LOGIC = re.compile(r"(?m)^\s*(func|cond|loop|while|for|repeat)\b|\bpush\s*\(")
LONG_STRING = re.compile(r"\"[^\"]{%d,}\"" % args.max_string_len)


def data_fraction(bs):
    """Share of characters sitting inside brackets — a data table, not logic.

    Depth-counted rather than regex-matched because the blobs are nested:
    FLAVOR_MATRIX = [[0, 0, 1, ...], [1, 0, 0, ...], ...] is one statement and
    19,710 tokens, and no line-based heuristic sees anything wrong with it.
    """
    inside = depth = 0
    for ch in bs:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth = max(0, depth - 1)
        elif depth:
            inside += 1
    return inside / max(1, len(bs))


def reject(row):
    """Why this row is not training data, or None to keep it."""
    bs = row["bs"]
    if row.get("tokens", 0) < args.min_tokens:
        return "too small"
    if data_fraction(bs) > args.max_data_frac:
        return "data blob"
    if LONG_STRING.search(bs):
        return "giant string literal"
    if not LOGIC.search(bs):
        return "no logic"
    return None


def main():
    if not os.path.exists(args.src):
        raise SystemExit(f"no corpus at {args.src}")

    dropped = collections.Counter()
    dropped_rows = collections.Counter()
    seen = set()
    kept = kept_tokens = total = total_tokens = 0
    n_val = val_tokens = 0
    if args.stats:
        out = out_val = None
    elif args.val_frac > 0:
        out, out_val = open(OUT_TRAIN, "w"), open(OUT_VAL, "w")
    else:
        out, out_val = open(OUT, "w"), None

    with open(args.src) as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            total += 1
            total_tokens += row.get("tokens", 0)

            why = reject(row)
            if why is None:
                # The Stack is deduplicated by file, but the same trivial program
                # reaches identical BrittainScript from different sources — 27
                # separate files translated to push("Hello World").
                key = re.sub(r"\s+", " ", row["bs"]).strip()
                if key in seen:
                    why = "duplicate output"
                else:
                    seen.add(key)

            if why is not None:
                dropped[why] += row.get("tokens", 0)
                dropped_rows[why] += 1
                continue

            kept += 1
            kept_tokens += row.get("tokens", 0)
            held = out_val is not None and is_val(row)
            if held:
                n_val += 1
                val_tokens += row.get("tokens", 0)
            if out:
                (out_val if held else out).write(json.dumps(row) + "\n")
    for handle in (out, out_val):
        if handle:
            handle.close()

    print(f"read   {total:,} rows / {total_tokens:,} tokens  from {args.src}")
    print("\ndropped:")
    for why, n in dropped.most_common():
        print(f"  {why:<22}{dropped_rows[why]:>8,} rows  {n:>11,} tokens "
              f"({100*n/max(1,total_tokens):5.1f}%)")
    print(f"\nkept   {kept:,} rows / {kept_tokens:,} tokens "
          f"({100*kept_tokens/max(1,total_tokens):.1f}% of tokens, "
          f"{100*kept/max(1,total):.1f}% of rows)")
    if args.stats:
        return
    if args.val_frac > 0:
        print(f"  train {kept-n_val:,} rows / {kept_tokens-val_tokens:,} tokens "
              f"-> {OUT_TRAIN}")
        print(f"  val   {n_val:,} rows / {val_tokens:,} tokens -> {OUT_VAL}")
    else:
        print(f"wrote  {OUT}")


if __name__ == "__main__":
    main()
