# -*- coding: utf-8 -*-
"""Where did "made by Luke McLaren" come from?

Run 2 gets its own name right and its maker wrong. The suspect is the home
directory that runs through the trajectory corpus:

    /Users/<owner-username>/...

Split that and you get luke / mclaren / brittain, and the model appears to have
read the middle of it as a surname.

Context and target are very different risks here. The loss is masked to the
target turn, so a path sitting in a tool result teaches nothing directly. A
path the assistant WRITES -- in a tool call argument, or in prose -- is trained
on, token by token. This counts both.
"""
import collections
import json
import re

MIX = "/home/lukeb/brittain4/data/train_mix.jsonl"
HOME = re.compile(r"<owner-username>", re.I)
MCLAREN = re.compile(r"mclaren", re.I)

rows = [json.loads(line) for line in open(MIX, encoding="utf-8") if line.strip()]

context_only = target_hit = 0
by_kind = collections.Counter()
samples = []

for row in rows:
    target = row["target"]
    target_text = (target.get("content") or "") + json.dumps(target.get("tool_calls") or [])
    context_text = json.dumps(row.get("messages") or [], ensure_ascii=False)

    in_target = bool(HOME.search(target_text))
    in_context = bool(HOME.search(context_text))

    if in_target:
        target_hit += 1
        by_kind[row["kind"]] += 1
        if len(samples) < 3:
            found = HOME.search(target_text)
            start = max(0, found.start() - 40)
            samples.append(target_text[start:found.end() + 40].replace("\n", " "))
    elif in_context:
        context_only += 1

print("of %d examples:" % len(rows))
print("  path in the TARGET (trained on)   : %d" % target_hit)
print("  path only in the context          : %d" % context_only)
print("\ntargets by slice:")
for kind, count in by_kind.most_common():
    print("  %-14s %d" % (kind, count))

for text in samples:
    print("\n  ...%s..." % text)

# Against that, how much says the maker's actual name?
maker = sum(1 for r in rows
            if re.search(r"Luke Brittain", (r["target"].get("content") or ""), re.I))
print("\ntargets that say 'Luke Brittain': %d" % maker)
print("targets containing 'mclaren'     : %d"
      % sum(1 for r in rows
            if MCLAREN.search((r["target"].get("content") or "")
                              + json.dumps(r["target"].get("tool_calls") or []))))
