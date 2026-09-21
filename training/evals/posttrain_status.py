# -*- coding: utf-8 -*-
"""Where the training mix actually stands, read off disk rather than recalled.

The last status I gave quoted 1,691 harvested examples; the file on disk held
1,121 and the larger harvest had never been written. So this reports what is
present, not what was once produced.
"""
import glob
import json
import os
from collections import Counter

DATA = "/home/lukeb/brittain4/data"


def count_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return None


def mtime(path):
    try:
        import datetime
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
    except OSError:
        return "-"


print("=" * 74)
print("ON DISK")
print("=" * 74)
for name in ("trajectories.jsonl", "identity_sft.jsonl", "bs_corpus.jsonl",
             "eval_driver.jsonl", "eval_general.jsonl", "eval_bs.jsonl"):
    path = os.path.join(DATA, name)
    n = count_lines(path)
    print("  %-22s %-9s  last written %s"
          % (name, ("%d rows" % n) if n is not None else "MISSING", mtime(path)))

chats = [p for p in glob.glob(os.path.join(DATA, "chats", "**", "*.json"), recursive=True)
         if os.path.basename(p) != "index.json"]
ids = set()
for p in chats:
    try:
        ids.add(json.load(open(p, encoding="utf-8")).get("id"))
    except Exception:
        pass
print("  %-22s %d files, %d unique chats" % ("chats/", len(chats), len(ids)))

excl = os.path.join(DATA, "excluded_chats.json")
if os.path.exists(excl):
    print("  %-22s %s" % ("excluded_chats.json", ", ".join(json.load(open(excl))) or "empty"))

print()
print("=" * 74)
print("CONTAMINATION: do eval items appear in the training pool?")
print("=" * 74)
# The eval sets were built from the same harvest the mix draws on. If the same
# chat_id shows up in both, every checkpoint comparison is scoring memorisation.
try:
    train_ids = set()
    for line in open(os.path.join(DATA, "trajectories.jsonl"), encoding="utf-8"):
        train_ids.add(json.loads(line).get("chat_id"))
    eval_ids = set()
    for line in open(os.path.join(DATA, "eval_driver.jsonl"), encoding="utf-8"):
        eval_ids.add(json.loads(line).get("chat_id"))
    overlap = train_ids & eval_ids
    print("  training chats : %d" % len(train_ids))
    print("  driver-eval chats: %d" % len(eval_ids))
    print("  OVERLAP        : %d chats (%.0f%% of the eval set)"
          % (len(overlap), 100.0 * len(overlap) / max(1, len(eval_ids))))
    if overlap:
        print("  -> the driver eval scores items the adapter would be trained on")
except Exception as error:
    print("  could not compare: %s" % error)

print()
print("=" * 74)
print("IDENTITY DATA vs WHAT THE PRODUCT NOW SAYS")
print("=" * 74)
try:
    rows = [json.loads(l) for l in open(os.path.join(DATA, "identity_sft.jsonl"), encoding="utf-8")]
    names = Counter()
    for r in rows:
        text = r["messages"][-1]["content"]
        for label in ("BRITTAIN-4", "Brittain 4", "BRITTAIN"):
            if label in text:
                names[label] += 1
                break
    print("  %d examples; name used in answers: %s" % (len(rows), dict(names)))
    print("  by kind: %s" % dict(Counter(r["kind"] for r in rows)))
except Exception as error:
    print("  could not read: %s" % error)
