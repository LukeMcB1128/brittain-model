# -*- coding: utf-8 -*-
"""Identify the "add caching" chat, and profile long no-ask_user chains.

Grepping the whole trajectory for "cache" matched 20 chats -- the word appears
in tool results and in source files being read, so it identifies nothing. What
is wanted is the chat where the USER asked for caching work.

The second half matters regardless: the actual objection to that trajectory was
a long unbroken run of tool calls with no ask_user. That is a measurable
property, so it is measured for every chat rather than resting on one
remembered example.
"""
import glob
import json
import os

ROOTS = glob.glob("/mnt/c/Users/lukeb/AppData/Local/Temp/claude/"
                  "C--Coding-brittain-model/*/scratchpad/chats_*")


def load(path):
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None, None
    if isinstance(data, list):
        return {"id": os.path.basename(path)}, data
    msgs = data.get("messages") or (data.get("chat") or {}).get("messages") or []
    return data, msgs


def profile(path):
    data, msgs = load(path)
    if not msgs:
        return None
    users = [m.get("content") for m in msgs
             if m.get("role") == "user" and isinstance(m.get("content"), str)]
    calls = sum(len(m.get("tool_calls") or []) for m in msgs)
    asked = sum(1 for m in msgs for c in (m.get("tool_calls") or [])
                if ((c.get("function") or {}).get("name") or c.get("name")) == "ask_user")
    # Longest run of tool calls with no user turn breaking it up.
    longest = run = 0
    for m in msgs:
        if m.get("tool_calls"):
            run += len(m["tool_calls"])
            longest = max(longest, run)
        elif m.get("role") == "user":
            run = 0
    return {"id": data.get("id") or os.path.basename(path),
            "users": users, "tool_calls": calls,
            "ask_user": asked, "longest_run": longest}


seen = {}
for root in ROOTS:
    for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        info = profile(path)
        if info:
            seen[info["id"]] = info

print("scanned %d unique chats\n" % len(seen))

print("=" * 74)
print("chats where a USER message mentions caching")
print("=" * 74)
hits = [i for i in seen.values() if any("cach" in t.lower() for t in i["users"])]
if not hits:
    print("  none -- no user turn ever phrased the request that way")
for info in sorted(hits, key=lambda i: -i["tool_calls"]):
    line = next(t for t in info["users"] if "cach" in t.lower())
    print("  id=%s calls=%d ask_user=%d longest_run=%d"
          % (info["id"], info["tool_calls"], info["ask_user"], info["longest_run"]))
    print("    %s" % " ".join(line.split())[:150])

print()
print("=" * 74)
print("longest unbroken tool runs, chats with NO ask_user at all (top 12)")
print("=" * 74)
worst = sorted((i for i in seen.values() if i["ask_user"] == 0),
               key=lambda i: -i["longest_run"])[:12]
for info in worst:
    opening = " ".join((info["users"][0] if info["users"] else "").split())[:86]
    print("  run=%3d calls=%3d id=%s" % (info["longest_run"], info["tool_calls"], info["id"]))
    print("      %s" % opening)
