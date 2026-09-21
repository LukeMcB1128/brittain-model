# -*- coding: utf-8 -*-
"""What personal data is in the training mix?

This decides whether the adapter can be published. Redaction at harvest covered
credential SHAPES -- keys, tokens, private keys -- because those were the
liability in a stored transcript. It did not touch ordinary personal detail,
because for training on your own machine there was no reason to.

Publishing changes that. A fine-tuned model can reproduce what it was trained
on, and the trajectory slice is real sessions from a real machine.
"""
import collections
import json
import re

MIX = "/home/lukeb/brittain4/data/train_mix.jsonl"

PATTERNS = {
    "posix home path": re.compile(r"/Users/[A-Za-z0-9._-]+"),
    "windows user path": re.compile(r"C:\\Users\\[A-Za-z0-9._-]+", re.I),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "personal name": re.compile(r"<owner-username>|luke[._]brittain", re.I),
    "local IP / host": re.compile(r"\b(?:192\.168|10\.\d+|localhost:\d+)\b"),
    "repo path": re.compile(r"BrittainCode-Projects|brittain-model|Downloads/Coding"),
}

rows = [json.loads(line) for line in open(MIX, encoding="utf-8") if line.strip()]
hits = collections.Counter()
by_kind = collections.defaultdict(collections.Counter)
samples = {}

for row in rows:
    text = json.dumps(row, ensure_ascii=False)
    for name, pattern in PATTERNS.items():
        found = pattern.search(text)
        if found:
            hits[name] += 1
            by_kind[row["kind"]][name] += 1
            samples.setdefault(name, found.group(0)[:64])

print("of %d training examples, how many contain:" % len(rows))
for name, count in hits.most_common():
    print("  %-20s %5d  (%4.1f%%)   e.g. %s"
          % (name, count, 100.0 * count / len(rows), samples[name]))

print("\nby slice:")
for kind in sorted(by_kind):
    total = sum(1 for r in rows if r["kind"] == kind)
    worst = by_kind[kind].most_common(2)
    print("  %-14s %4d examples   %s" % (
        kind, total, ", ".join("%s x%d" % (n, c) for n, c in worst) or "clean"))
