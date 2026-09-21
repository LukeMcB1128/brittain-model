# -*- coding: utf-8 -*-
"""Where does the clipped conversational register come from?

Live web chat, search working, answer correct -- and then:

    "Thanks g"                 -> "Good."
    "wdym good"                -> "I meant I have the answer - no problem."
    "why you acting so strange"-> "I'm not. Just being concise."
    "I get that"               -> "Good."

That is not brevity, it is a register. Register is global: a model does not
learn "be brief about yourself", it learns "be brief". So the question is which
slice of the mix is short, and how short.

Reports the distribution of TARGET lengths per kind, and how much of each kind
is very short, since it is the mass of curt targets that sets the tone rather
than the average.
"""
import collections
import json
import statistics

MIX = "/home/lukeb/brittain4/data/train_mix.jsonl"
rows = [json.loads(l) for l in open(MIX, encoding="utf-8") if l.strip()]

lengths = collections.defaultdict(list)
prose_free = collections.Counter()
totals = collections.Counter()

for row in rows:
    kind = row["kind"]
    target = row["target"]
    content = (target.get("content") or "").strip()
    totals[kind] += 1
    if target.get("tool_calls") and not content:
        prose_free[kind] += 1
        continue                      # a bare tool call has no register
    lengths[kind].append(len(content))

print("%-12s %7s %8s %8s %8s %8s %10s" % (
    "kind", "n", "median", "mean", "p10", "p90", "<40 chars"))
print("-" * 68)
for kind in ("trajectory", "general", "identity", "restraint"):
    values = sorted(lengths[kind])
    if not values:
        continue
    short = sum(1 for v in values if v < 40)
    print("%-12s %7d %8d %8d %8d %8d %6d (%2.0f%%)" % (
        kind, len(values), statistics.median(values),
        statistics.mean(values), values[len(values) // 10],
        values[int(len(values) * 0.9)], short, 100.0 * short / len(values)))

print("\nprose-free targets (a tool call with no words):")
for kind in ("trajectory", "general", "identity", "restraint"):
    if totals[kind]:
        print("  %-12s %4d of %4d  (%2.0f%%)" % (
            kind, prose_free[kind], totals[kind],
            100.0 * prose_free[kind] / totals[kind]))

# The very short replies are the ones that set the tone. Show some.
print("\nshortest non-empty targets in the mix:")
shown = 0
for row in sorted(rows, key=lambda r: len((r["target"].get("content") or "").strip())):
    content = (row["target"].get("content") or "").strip()
    if not content or row["target"].get("tool_calls"):
        continue
    print("  [%-10s] %r" % (row["kind"], content[:90]))
    shown += 1
    if shown >= 12:
        break
