"""Compare two eval runs on the same examples, with a test that fits the design.

    python3 compare_runs.py a.json b.json

The two runs score the SAME examples, so the comparison is paired and the right
test is McNemar's on the discordant pairs -- how often A succeeded where B
failed, versus the reverse. Comparing two independent proportions throws away
the pairing and overstates the uncertainty.

Paired BY POSITION: both runs read the same eval file in the same order, so
index i is the same example in each. An earlier version keyed on
(chat_id, reference), which collapses the many examples sharing both -- it
quietly paired 43 of 97 and discarded the rest.

n is small. A couple of points of difference is noise, and saying otherwise on
this sample size is how a project talks itself into a conclusion it cannot
support.
"""
import json
import math
import sys

METRICS = ("emitted_call", "name_valid", "schema_valid", "name_match")


def outcomes(path):
    """Per-example booleans, recomputed so both runs are scored identically."""
    d = json.load(open(path, encoding="utf-8"))
    per = []
    for r in d["records"]:
        emitted = bool(r["emitted"])
        valid_name = emitted and not r.get("invalid_name")
        per.append({
            "reference": r["reference"],
            "emitted_call": emitted,
            "name_valid": valid_name,
            "schema_valid": valid_name and not r.get("schema_error"),
            "name_match": emitted and r["emitted"][0] == r["reference"],
        })
    return d["summary"]["label"], d["summary"], per


def mcnemar(b, c):
    """Two-sided exact binomial p over the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


la, sa, pa = outcomes(sys.argv[1])
lb, sb, pb = outcomes(sys.argv[2])
assert len(pa) == len(pb), "different example counts: %d vs %d" % (len(pa), len(pb))
bad = [i for i in range(len(pa)) if pa[i]["reference"] != pb[i]["reference"]]
assert not bad, "runs are not in the same order, first at %s" % bad[:5]

n = len(pa)
print("A: %s" % la)
print("B: %s" % lb)
print("paired examples: %d\n" % n)

print("%-16s %8s %8s %8s   %s"
      % ("metric", "A", "B", "diff", "discordant (A only / B only), p"))
for m in METRICS:
    a_only = sum(1 for i in range(n) if pa[i][m] and not pb[i][m])
    b_only = sum(1 for i in range(n) if pb[i][m] and not pa[i][m])
    a_rate = 100 * sum(pa[i][m] for i in range(n)) / n
    b_rate = 100 * sum(pb[i][m] for i in range(n)) / n
    p = mcnemar(a_only, b_only)
    verdict = "significant" if p <= 0.05 else "not distinguishable"
    print("%-16s %7.1f%% %7.1f%% %+7.1f    %2d / %2d, p=%.3f  %s"
          % (m, a_rate, b_rate, a_rate - b_rate, a_only, b_only, p, verdict))

print("\nas reported by each run:")
print("  A:", json.dumps(sa["rates"]))
print("  B:", json.dumps(sb["rates"]))
