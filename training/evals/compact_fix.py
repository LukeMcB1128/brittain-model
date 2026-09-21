# -*- coding: utf-8 -*-
"""Validate the proposed /compact fix at the boundary.

Established by compact_limits.py:
  - The model compacts correctly (5/5 headings) whenever the request fits.
  - It fails with HTTP 400 when prompt + max_tokens > 32768.
  - The 55 tool schemas cost 8,472 tokens and are dead weight on a
    summarisation request: with tools, compaction breaks from ~22k of
    conversation; without them, it survives to ~28k.

So the fix is budget arithmetic, not prompting. Two parts to check:
  A. Dropping tools and sizing max_tokens to the space actually left.
  B. Whether a summary squeezed into a small budget is still usable, since the
     real records run ~1,900 tokens. If not, compaction of a nearly-full window
     needs two passes rather than a smaller budget.
"""
import glob
import json
import os
import urllib.request

BASE = "http://localhost:11435/v1/chat/completions"
KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()
LIMIT = 32768

COMPACT = ("Summarize the conversation so far. Begin with the line "
           "\"Summary of the conversation so far:\" and use exactly these "
           "headings: GOAL, CONSTRAINTS, DECISIONS, STATE, NEXT. Capture "
           "everything needed to continue the work without the original "
           "messages.")

pool = []
for path in glob.glob("/home/lukeb/brittain4/data/chats/**/*.json", recursive=True):
    if os.path.basename(path) == "index.json":
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    for m in data.get("conversation") or []:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"].strip():
            pool.append({"role": m["role"], "content": m["content"]})
    if len(pool) > 400:
        break


def conversation_of(target_tokens):
    msgs, total, want = [], 0, target_tokens * 4
    cap = max(2000, want // 12)
    for m in pool:
        if total >= want:
            break
        chunk = m["content"][:min(cap, want - total)]
        if not chunk.strip():
            continue
        msgs.append({"role": "user" if len(msgs) % 2 == 0 else "assistant",
                     "content": chunk})
        total += len(chunk)
    if msgs[-1]["role"] != "assistant":
        msgs.append({"role": "assistant", "content": "Understood, continuing."})
    return msgs


def call(messages, max_tokens):
    body = {"model": "brittain4", "temperature": 0.7, "max_tokens": max_tokens,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.load(r)
    except urllib.error.HTTPError as error:
        return None, "HTTP %d" % error.code, 0
    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    return content, choice.get("finish_reason"), data["usage"]["prompt_tokens"]


def score(content):
    return sum(1 for h in ("GOAL", "CONSTRAINTS", "DECISIONS", "STATE", "NEXT")
               if content and h in content)


print("=" * 96)
print("A. no tools, max_tokens sized to what is left (reserve 256 for the template)")
print("=" * 96)
for size in (26000, 28000, 30000):
    conv = conversation_of(size) + [{"role": "user", "content": COMPACT}]
    # One cheap probe to learn the true prompt size, then size the budget.
    _, _, prompt_tokens = call(conv, 16)
    if not prompt_tokens:
        print("  conv~%-6s could not even be measured" % size)
        continue
    budget = max(256, LIMIT - prompt_tokens - 256)
    content, finish, _ = call(conv, budget)
    print("  conv~%-6s prompt=%-6d budget=%-5d finish=%-8s headings=%d/5 reply=%d chars"
          % (size, prompt_tokens, budget, finish, score(content),
             len(content or "")))

print()
print("=" * 96)
print("B. two-pass when even that is too tight: halve, summarise each, merge")
print("=" * 96)
conv = conversation_of(30000)
half = len(conv) // 2
parts = []
for i, chunk in enumerate((conv[:half], conv[half:])):
    msgs = chunk + [{"role": "user", "content": COMPACT}]
    _, _, prompt_tokens = call(msgs, 16)
    content, finish, _ = call(msgs, max(256, min(1400, LIMIT - prompt_tokens - 256)))
    parts.append(content or "")
    print("  half %d: prompt=%-6d finish=%-8s headings=%d/5 reply=%d chars"
          % (i + 1, prompt_tokens, finish, score(content), len(content or "")))

merge = [{"role": "user", "content":
          "Here are two partial summaries of one conversation, in order.\n\n"
          "--- PART 1 ---\n" + parts[0] + "\n\n--- PART 2 ---\n" + parts[1]
          + "\n\n" + COMPACT}]
content, finish, prompt_tokens = call(merge, 2048)
print("  merged : prompt=%-6d finish=%-8s headings=%d/5 reply=%d chars"
      % (prompt_tokens, finish, score(content), len(content or "")))
print("=" * 96)
if content:
    print("\n--- merged summary, opening ---")
    print(content[:400])
