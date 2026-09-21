# -*- coding: utf-8 -*-
"""What does a working compaction look like in the corpus?

Brittain Code's /compact prompt lives in its own repo, which is not here. But
six chats carry a compactionRecord written by whatever model was driving at the
time, so the expected OUTPUT is available even though the instruction is not.
Reconstructing the target from the output is enough to test whether BRITTAIN-4
can produce it.
"""
import glob
import json
import os
import re
from collections import Counter

records = []
for path in glob.glob("/home/lukeb/brittain4/data/chats/**/*.json", recursive=True):
    if os.path.basename(path) == "index.json":
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    for msg in data.get("conversation") or []:
        if msg.get("compactionRecord") and isinstance(msg.get("content"), str):
            records.append((data.get("id"), data.get("model"), msg["content"]))

# The same chat appears in several zips; keep one copy of each.
unique = {}
for chat_id, model, content in records:
    unique[chat_id] = (model, content)

print("unique compaction records: %d\n" % len(unique))

heading = re.compile(r"^[#*\s]*([A-Z][A-Z /&_-]{2,})\s*[:*]*\s*$")
inline = re.compile(r"^[#*\s]*([A-Z][A-Z /&_-]{2,}):")
counts = Counter()
lengths = []

for chat_id, (model, content) in unique.items():
    lengths.append(len(content))
    for line in content.split("\n"):
        s = line.strip()
        m = heading.match(s) or inline.match(s)
        if m:
            counts[m.group(1).strip()] += 1
    print("  %-32s model=%-22s %6d chars  opens: %s"
          % (chat_id, str(model)[:22], len(content),
             " ".join(content.split())[:60]))

print("\nheadings used across records:")
for name, n in counts.most_common(12):
    print("  %-22s %d" % (name, n))
print("\nsummary length: min %d  median %d  max %d chars"
      % (min(lengths), sorted(lengths)[len(lengths) // 2], max(lengths)))

sample = sorted(unique.items(), key=lambda kv: len(kv[1][1]))[len(unique) // 2]
print("\n--- median record, first 1200 chars ---")
print(sample[1][1][:1200])
