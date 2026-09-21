"""Build the BrittainScript eval: translate Python, then check it actually runs.

    python3 build_bs_eval.py --in bs_corpus.jsonl --out eval_bs.jsonl

WHY FUNCTIONAL EQUIVALENCE AND NOT SIMILARITY
Comparing generated BrittainScript to the reference text rewards imitation and
punishes a correct program written differently. Running both and comparing what
they print is the real question, and it is automatic.

Only programs that actually PRINT are usable, so this keeps rows whose
reference produces non-empty stdout. push() is the output function; 66% of the
kept corpus uses it.

THE SPLIT MATCHES filter_bs.py
Content hash, not position or shuffle, so a program stays on one side as the
corpus grows -- and so anything measured here was never in the training file.
Held-out means hash-based val, same rule, so the two pipelines cannot disagree.

WHAT THIS DOES NOT COVER
The corpus is transpiled Python from The Stack, and only 10% of it uses
pyimport, against 6 uses in each of the hand-written .bs files. So this eval
measures standalone algorithmic BrittainScript. Fluency in the pyimport-heavy
style needs its own examples and its own eval.
"""
import argparse
import collections
import json
import random
import re
import sys

sys.path.insert(0, "/tmp")
from bs_check import check                                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--val-frac", type=float, default=0.05)
ap.add_argument("--max-py-chars", type=int, default=1200,
                help="keep prompts small; long ones crowd the eval budget")
args = ap.parse_args()

LOGIC = re.compile(r"(?m)^\s*(func|cond|loop|while|for|repeat)\b|\bpush\s*\(")
LONG_STRING = re.compile(r"\"[^\"]{400,}\"")


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


def kept(row):
    bs = row["bs"]
    return (row.get("tokens", 0) >= 12
            and data_fraction(bs) <= 0.5
            and not LONG_STRING.search(bs)
            and LOGIC.search(bs))


def is_val(row):
    return int(row["hash"], 16) % 10_000 < args.val_frac * 10_000


rows = []
with open(args.src, encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if kept(r) and is_val(r) and r.get("py") and len(r["py"]) <= args.max_py_chars:
            rows.append(r)

print("held-out candidates: %d" % len(rows))
random.Random(11).shuffle(rows)

out, stats = [], collections.Counter()
for r in rows:
    if len(out) >= args.n:
        break
    res = check(r["bs"], timeout=20)
    if not res["valid"]:
        stats["reference does not run"] += 1
        continue
    expected = (res["output"] or "").strip()
    if not expected:
        stats["reference prints nothing"] += 1
        continue
    stats["usable"] += 1
    out.append({
        "task": "bs_translate",
        "hash": r["hash"],
        "python": r["py"],
        "reference_bs": r["bs"],
        "expected_output": expected,
        "uses_pyimport": "pyimport" in r["bs"],
    })

with open(args.out, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")

print()
for k, v in stats.most_common():
    print("  %-26s %d" % (k, v))
print("\neval items      : %d" % len(out))
print("use pyimport    : %d" % sum(r["uses_pyimport"] for r in out))
print("median py chars : %d"
      % sorted(len(r["python"]) for r in out)[len(out) // 2] if out else 0)
print("wrote %s" % args.out)
