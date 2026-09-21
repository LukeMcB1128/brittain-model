"""Show the checkpoint comparison as counts, not percentages.

A percentage on n=80 hides that a 1.2-point gap is a single question. Raw
counts make the size of each difference obvious, and this comparison exists to
decide whether one checkpoint is worse than another -- a decision that should
not rest on one item going the other way.
"""
import collections
import json
import sys

def load(path):
    d = json.load(open(path, encoding="utf-8"))
    per = collections.defaultdict(lambda: [0, 0])
    for r in d["records"]:
        per[r["task"]][0] += bool(r["ok"])
        per[r["task"]][1] += 1
    return d["summary"]["label"], per

la, a = load(sys.argv[1])
lb, b = load(sys.argv[2])

print("A = %s" % la)
print("B = %s" % lb)
print()
print("%-16s %14s %14s %8s" % ("task", "A", "B", "delta"))
tasks = sorted(set(a) | set(b))
for t in tasks:
    ac, an = a[t]
    bc, bn = b[t]
    print("%-16s %6d/%-3d %4.1f%% %6d/%-3d %4.1f%% %+6d"
          % (t, ac, an, 100 * ac / max(an, 1), bc, bn, 100 * bc / max(bn, 1), bc - ac))

caps = [t for t in ("gsm8k", "mmlu", "arc", "humaneval") if t in a]
ac = sum(a[t][0] for t in caps); an = sum(a[t][1] for t in caps)
bc = sum(b[t][0] for t in caps); bn = sum(b[t][1] for t in caps)
print("%-16s %6d/%-3d %4.1f%% %6d/%-3d %4.1f%% %+6d"
      % ("CAPABILITY", ac, an, 100 * ac / an, bc, bn, 100 * bc / bn, bc - ac))

# Where they actually disagree, item by item.
print("\nitems where exactly one got it right:")
ra = json.load(open(sys.argv[1], encoding="utf-8"))["records"]
rb = json.load(open(sys.argv[2], encoding="utf-8"))["records"]
if len(ra) == len(rb):
    flips = collections.Counter()
    for x, y in zip(ra, rb):
        if x["ok"] != y["ok"]:
            flips[(x["task"], "B only" if y["ok"] else "A only")] += 1
    for (t, who), n in sorted(flips.items()):
        print("   %-16s %-8s %d" % (t, who, n))
    total_flips = sum(flips.values())
    print("\n   %d of %d items disagree (%.1f%%)"
          % (total_flips, len(ra), 100 * total_flips / len(ra)))
else:
    print("   (different lengths: %d vs %d)" % (len(ra), len(rb)))
