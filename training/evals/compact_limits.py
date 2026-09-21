# -*- coding: utf-8 -*-
"""At what conversation size does /compact stop working?

The first repro compacted a 13.7k-token conversation successfully in 5 of 6
configurations, so the model can do the task. That means the repro was not
reproducing the real conditions.

The real conditions: /compact is invoked when the context is nearly exhausted
-- that is the whole point of it -- and Brittain Code sends all 55 tool schemas
on every request, which cost 7,725-8,073 tokens before a single message. So the
request that asks for a summary is the largest request in the session, and it
has to fit a ~1,900-token reply on top of that.

This sweeps conversation size against tools on/off and reports the exact
failure, so the fix can be sized rather than guessed.
"""
import glob
import json
import os
import urllib.request

BASE = "http://localhost:11435/v1/chat/completions"
KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()
LIMIT = 32768

COMPACT_INSTRUCTION = (
    "Summarize the conversation so far. Begin with the line "
    "\"Summary of the conversation so far:\" and use exactly these headings: "
    "GOAL, CONSTRAINTS, DECISIONS, STATE, NEXT. Capture everything needed to "
    "continue the work without the original messages.")

defs = json.load(open("/home/lukeb/brittain4/data/tool_defs.json",
                      encoding="utf-8"))
TOOL_TOKENS = len(json.dumps(defs)) // 4          # rough, but the right order

# Build one long realistic conversation by concatenating real chats.
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
    """Alternating turns totalling roughly target_tokens.

    Messages are truncated to a slice of the budget: single corpus messages run
    to tens of thousands of characters, so appending whole ones overshot every
    target and made all five sizes identical.
    """
    msgs, total, want = [], 0, target_tokens * 4
    cap = max(2000, want // 12)
    for m in pool:
        if total >= want:
            break
        chunk = m["content"][:min(cap, want - total)]
        if not chunk.strip():
            continue
        role = "user" if len(msgs) % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": chunk})
        total += len(chunk)
    if not msgs or msgs[-1]["role"] != "assistant":
        msgs.append({"role": "assistant", "content": "Understood, continuing."})
    return msgs


def attempt(conv_tokens, with_tools, max_tokens=2048):
    conv = conversation_of(conv_tokens)
    body = {"model": "brittain4", "temperature": 0.7, "max_tokens": max_tokens,
            "messages": conv + [{"role": "user", "content": COMPACT_INSTRUCTION}],
            "chat_template_kwargs": {"enable_thinking": False}}
    if with_tools:
        body["tools"] = defs
        body["tool_choice"] = "auto"
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.load(r)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("message") or json.loads(detail).get("error", {}).get("message") or detail
        except Exception:
            pass
        return ("HTTP %d" % error.code, " ".join(str(detail).split())[:150])
    except Exception as error:
        return ("ERROR", str(error)[:120])
    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    headings = sum(1 for h in ("GOAL", "CONSTRAINTS", "DECISIONS", "STATE", "NEXT") if h in content)
    prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
    return ("%s %d/5 headings" % (choice.get("finish_reason"), headings),
            "prompt=%d tokens, reply=%d chars" % (prompt_tokens, len(content)))


print("55 tool schemas cost roughly %d tokens before any message" % TOOL_TOKENS)
print("server max_model_len = %d\n" % LIMIT)
print("=" * 100)
print("%-11s %-7s %-26s %s" % ("conv tokens", "tools", "result", "detail"))
print("=" * 100)
for size in (8000, 16000, 22000, 26000, 30000):
    for with_tools in (False, True):
        verdict, detail = attempt(size, with_tools)
        print("%-11s %-7s %-26s %s"
              % (size, "yes" if with_tools else "no", verdict, detail))
print("=" * 100)
