# -*- coding: utf-8 -*-
"""Reproduce the /compact failure against BRITTAIN-4.

Brittain Code's compaction prompt lives in its own repo, so the instruction
here is RECONSTRUCTED from the six compaction records in the corpus: every one
opens "Summary of the conversation so far:" and uses exactly GOAL, CONSTRAINTS,
DECISIONS, STATE, NEXT. Median output is 7,364 characters, roughly 1,900
tokens -- which is the reason for the first hypothesis below.

Hypotheses, in the order they are worth testing:
  H1 BUDGET. Thinking is on by default through the API, and the app sends some
     max_tokens. If the trace eats the budget, content comes back empty or
     truncated. This exact failure has already appeared on this project once,
     where a 200-token cap left content empty and the trace full.
  H2 TOOL CALL. The app sends its 55 tools on every request. The model may
     answer a summarisation instruction with a tool call instead of prose.
  H3 FORMAT. It summarises, but without the five headings, so the app's parser
     rejects it.

Each is distinguishable from the others by what comes back, so one run over a
grid of (thinking, budget, tools) separates them.
"""
import glob
import json
import os
import urllib.request

BASE = "http://localhost:11435/v1/chat/completions"
KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()

COMPACT_INSTRUCTION = (
    "Summarize the conversation so far. Begin with the line "
    "\"Summary of the conversation so far:\" and use exactly these headings: "
    "GOAL, CONSTRAINTS, DECISIONS, STATE, NEXT. Capture everything needed to "
    "continue the work without the original messages.")

defs = json.load(open("/home/lukeb/brittain4/data/tool_defs.json",
                      encoding="utf-8"))


def load_conversation(limit_chars=60000):
    """A real chat, trimmed to something that fits alongside the reply."""
    best = None
    for path in glob.glob("/home/lukeb/brittain4/data/chats/chats_v5/**/*.json", recursive=True):
        if os.path.basename(path) == "index.json":
            continue
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        msgs = []
        for m in data.get("conversation") or []:
            role, content = m.get("role"), m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                msgs.append({"role": role, "content": content})
        if len(msgs) >= 12:
            total = sum(len(m["content"]) for m in msgs)
            if best is None or abs(total - limit_chars) < abs(best[1] - limit_chars):
                best = (msgs, total, data.get("title"))
    msgs, total, title = best
    while sum(len(m["content"]) for m in msgs) > limit_chars and len(msgs) > 8:
        msgs.pop(0)
    if msgs[0]["role"] != "user":
        msgs.insert(0, {"role": "user", "content": "Let's continue."})
    return msgs, title


CONV, TITLE = load_conversation()
print("compacting: %s" % TITLE)
print("  %d messages, %d chars (~%d tokens)\n"
      % (len(CONV), sum(len(m["content"]) for m in CONV),
         sum(len(m["content"]) for m in CONV) // 4))


def run(label, thinking, max_tokens, with_tools):
    body = {"model": "brittain4", "temperature": 0.7,
            "max_tokens": max_tokens,
            "messages": CONV + [{"role": "user", "content": COMPACT_INSTRUCTION}]}
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
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
    except Exception as error:
        print("  %-34s ERROR %s" % (label, str(error)[:70]))
        return
    choice = data["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    calls = msg.get("tool_calls") or []
    headings = [h for h in ("GOAL", "CONSTRAINTS", "DECISIONS", "STATE", "NEXT")
                if h in content]
    verdict = ("TOOL CALL: " + calls[0]["function"]["name"] if calls
               else "EMPTY CONTENT" if not content.strip()
               else "OK" if len(headings) == 5
               else "PARTIAL (%d/5 headings)" % len(headings))
    print("  %-34s content=%5d reason=%5d finish=%-12s %s"
          % (label, len(content), len(reasoning),
             choice.get("finish_reason"), verdict))
    return content


print("=" * 92)
print("H1/H2/H3 grid")
print("=" * 92)
run("think ON,  2048 tok, tools", True, 2048, True)
run("think ON,  2048 tok, no tools", True, 2048, False)
run("think OFF, 2048 tok, tools", False, 2048, True)
run("think OFF, 2048 tok, no tools", False, 2048, False)
run("think ON,  4096 tok, no tools", True, 4096, False)
best = run("think OFF, 4096 tok, no tools", False, 4096, False)
print("=" * 92)
if best:
    print("\n--- opening of the best result ---")
    print(best[:500])
