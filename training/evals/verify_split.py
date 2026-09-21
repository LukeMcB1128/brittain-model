# -*- coding: utf-8 -*-
"""The check that matters: is the training file disjoint from the eval sets?

Not "did the splitter run" but "does any chat the adapter trains on also appear
in an eval". That is the condition that makes a checkpoint comparison
meaningless, and it is cheap to assert directly.
"""
import collections
import json
import os

DATA = "/home/lukeb/brittain4/data"


def chat_ids(path):
    ids = set()
    try:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if row.get("chat_id"):
                ids.add(row["chat_id"])
    except OSError:
        return None
    return ids


train = chat_ids(os.path.join(DATA, "trajectories_train.jsonl"))
held = chat_ids(os.path.join(DATA, "trajectories_heldout.jsonl"))

print("=" * 72)
print("train chats   : %d" % len(train))
print("held-out chats: %d" % len(held))
print("train ∩ heldout: %d  %s" % (len(train & held), "OK" if not train & held else "LEAK"))
print("=" * 72)

failures = 0
for name in ("eval_driver.jsonl", "eval_bs.jsonl"):
    path = os.path.join(DATA, name)
    ids = chat_ids(path)
    if ids is None:
        print("%-20s missing" % name)
        continue
    if not ids:
        print("%-20s no chat_id field (not harvested from chats)" % name)
        continue
    leak = ids & train
    status = "OK" if not leak else "LEAK of %d chats" % len(leak)
    if leak:
        failures += 1
    print("%-20s %3d chats, %3d also in TRAIN  -> %s" % (name, len(ids), len(leak), status))
    if leak:
        print("   %s" % ", ".join(sorted(leak)[:6]))

print()
print("VERDICT:", "training set is clean" if not failures else "CONTAMINATED")
