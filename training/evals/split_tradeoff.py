# -*- coding: utf-8 -*-
"""How much held-out tool coverage is actually purchasable?

A clean split costs coverage: 12 held-out chats exercised 40 of the 73 tools
the training side calls, so a driver eval built from them is blind to the rest.
Before spending training data to buy coverage, it is worth knowing the ceiling
-- a tool that appears in only one chat can be on one side or the other, never
both, at any split ratio.
"""
import collections
import hashlib
import json

SRC = "/home/lukeb/brittain4/data/trajectories_new.jsonl"
rows = [json.loads(line) for line in open(SRC, encoding="utf-8")]

by_chat = collections.defaultdict(list)
for row in rows:
    by_chat[row.get("chat_id")].append(row)


def tool_names(examples):
    names = set()
    for row in examples:
        for call in (row.get("target") or {}).get("tool_calls") or []:
            name = (call.get("function") or {}).get("name") or call.get("name")
            if name:
                names.add(name)
    return names


chats_per_tool = collections.Counter()
for chat_id, examples in by_chat.items():
    for name in tool_names(examples):
        chats_per_tool[name] += 1

only_one = [t for t, n in chats_per_tool.items() if n == 1]
print("distinct tools called anywhere : %d" % len(chats_per_tool))
print("tools appearing in only ONE chat: %d" % len(only_one))
print("  -> these can never be on both sides, whatever the ratio")
print("tools in >= 2 chats            : %d (the real ceiling for both-sides coverage)"
      % sum(1 for n in chats_per_tool.values() if n >= 2))
print()

print("=" * 78)
print("%-8s %-8s %-11s %-14s %s"
      % ("share", "chats", "examples", "held-out tools", "train tools lost entirely"))
print("=" * 78)
for share in (0.10, 0.15, 0.20, 0.25, 0.30):
    held, train = [], []
    for chat_id, examples in by_chat.items():
        digest = hashlib.sha256(str(chat_id).encode()).hexdigest()
        (held if int(digest[:8], 16) / 0xFFFFFFFF < share else train).append((chat_id, examples))
    held_tools = set().union(*[tool_names(e) for _, e in held]) if held else set()
    train_tools = set().union(*[tool_names(e) for _, e in train]) if train else set()
    # Tools the model would be trained on but never checked for held-out.
    unchecked = train_tools - held_tools
    # Worse: tools that leave the training set altogether.
    lost = held_tools - train_tools
    print("%-8.0f%% %-8d %-11d %-14d %d unchecked, %d absent from train"
          % (share * 100, len(held), sum(len(e) for _, e in held),
             len(held_tools), len(unchecked), len(lost)))
print("=" * 78)
print("""
Read: 'unchecked' is how many trained tools the held-out set cannot test.
'absent from train' is worse -- a tool the model never learns because its only
chats went to the held-out side.""")
