"""What is actually in the BrittainScript corpus, and does it match how you write it?

Three questions:

1. HOW MUCH SURVIVES FILTERING. Reimplements filter_bs.py's rules so the numbers
   here line up with the ones that built the training file: too small, data
   blob, giant string literal, no logic, duplicate output.

2. IS IT VALID. The corpus is py2bs output, so it is valid by construction --
   but the interpreter exits 0 on every program including broken ones, so
   "by construction" has never actually been checked. Runs a sample.

3. DOES IT LOOK LIKE YOUR CODE. This is the one that matters. The corpus is
   transpiled Python from The Stack; the two .bs files in brittain-model are
   hand-written and use pyimport six times each to reach into torch. If the
   corpus never uses pyimport, training on it produces a model fluent in
   transpiler dialect and not in the language as you write it.
"""
import argparse
import collections
import json
import random
import re
import sys

sys.path.insert(0, "/tmp")

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", required=True)
ap.add_argument("--sample-validity", type=int, default=150)
ap.add_argument("--min-tokens", type=int, default=12)
ap.add_argument("--max-data-frac", type=float, default=0.5)
ap.add_argument("--max-string-len", type=int, default=400)
args = ap.parse_args()

LOGIC = re.compile(r"(?m)^\s*(func|cond|loop|while|for|repeat)\b|\bpush\s*\(")
LONG_STRING = re.compile(r"\"[^\"]{%d,}\"" % args.max_string_len)


def data_fraction(bs):
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


rows = []
with open(args.src, encoding="utf-8") as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass

total_tokens = sum(r.get("tokens", 0) for r in rows)
print("records        : %d" % len(rows))
print("tokens (bs)    : %s" % f"{total_tokens:,}")
py_chars = sum(len(r.get("py") or "") for r in rows)
bs_chars = sum(len(r.get("bs") or "") for r in rows)
print("chars py / bs  : %s / %s" % (f"{py_chars:,}", f"{bs_chars:,}"))
print("paired         : %d have both py and bs"
      % sum(1 for r in rows if r.get("py") and r.get("bs")))

dropped, seen, kept = collections.Counter(), set(), []
drop_tokens = collections.Counter()
for r in rows:
    why = reject(r)
    if why is None:
        key = re.sub(r"\s+", " ", r["bs"]).strip()
        if key in seen:
            why = "duplicate output"
        else:
            seen.add(key)
    if why:
        dropped[why] += 1
        drop_tokens[why] += r.get("tokens", 0)
    else:
        kept.append(r)

print("\n--- filter (filter_bs.py rules) ---")
for why, n in dropped.most_common():
    print("  %-22s %6d rows  %9s tokens (%4.1f%%)"
          % (why, n, f"{drop_tokens[why]:,}", 100 * drop_tokens[why] / max(1, total_tokens)))
kept_tokens = sum(r.get("tokens", 0) for r in kept)
print("  %-22s %6d rows  %9s tokens (%4.1f%%)"
      % ("KEPT", len(kept), f"{kept_tokens:,}", 100 * kept_tokens / max(1, total_tokens)))

print("\n--- construct usage in the kept set ---")
CONSTRUCTS = ["pyimport", "push", "func", "cond", "while", "for", "loop",
              "repeat", "return", "end"]
for c in CONSTRUCTS:
    pat = re.compile(r"\b%s\b" % c)
    n = sum(1 for r in kept if pat.search(r["bs"]))
    print("  %-10s in %5d/%d programs (%5.1f%%)"
          % (c, n, len(kept), 100 * n / max(1, len(kept))))

print("\n--- the same, in the two hand-written files ---")
import glob
hand = glob.glob("/mnt/c/Coding/brittain-model/brittain_script/*.bs")
for path in hand:
    src = open(path, encoding="utf-8", errors="replace").read()
    counts = {c: len(re.findall(r"\b%s\b" % c, src)) for c in CONSTRUCTS}
    print("  %-22s %s" % (path.split("/")[-1],
                          {k: v for k, v in counts.items() if v}))

print("\n--- validity, on a random sample of the kept set ---")
try:
    from bs_check import check
except ImportError:
    print("  bs_check not importable; skipping")
    raise SystemExit

rng = random.Random(7)
sample = rng.sample(kept, min(args.sample_validity, len(kept)))
bad = collections.Counter()
ok = 0
for r in sample:
    res = check(r["bs"], timeout=20)
    if res["valid"]:
        ok += 1
    else:
        bad[res["reason"][:70]] += 1
print("  valid: %d/%d (%.1f%%)" % (ok, len(sample), 100 * ok / len(sample)))
for reason, n in bad.most_common(10):
    print("     [%d] %s" % (n, reason))
