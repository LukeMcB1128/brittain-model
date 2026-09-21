# -*- coding: utf-8 -*-
"""Drop the "add caching" trajectory from the training set.

Why: 29 consecutive tool calls with no ask_user anywhere in it. Trained on, it
teaches the model to keep going through long tool chains without ever checking
back with the person, which is the opposite of the behaviour the ask_user
examples were collected to reinforce.

Writes a backup first. This is the only copy of the harvest.
"""
import json
import shutil

SRC = "/home/lukeb/brittain4/data/trajectories.jsonl"
BACKUP = SRC + ".before-caching-drop"

rows = [json.loads(line) for line in open(SRC, encoding="utf-8")]
print("total examples: %d" % len(rows))

# Find it by content rather than by a remembered chat id.
def looks_like_caching(row):
    text = json.dumps(row).lower()
    return "caching" in text or "cache" in text


candidates = {}
for row in rows:
    if looks_like_caching(row):
        candidates.setdefault(row.get("chat_id"), 0)
        candidates[row["chat_id"]] += 1

print("\nchats mentioning cache/caching:")
for chat_id, count in sorted(candidates.items(), key=lambda kv: -kv[1]):
    turns = [r for r in rows if r.get("chat_id") == chat_id]
    calls = sum(len(r.get("target", {}).get("tool_calls") or []) for r in turns)
    asked = sum(1 for r in turns
                for c in (r.get("target", {}).get("tool_calls") or [])
                if (c.get("function", {}) or {}).get("name") == "ask_user"
                or c.get("name") == "ask_user")
    print("  %s  turns=%3d  tool_calls=%3d  ask_user=%d  (cache-matching turns=%d)"
          % (chat_id, len(turns), calls, asked, count))
