# -*- coding: utf-8 -*-
"""End-to-end smoke test: is the live server actually serving correctly?

Health 200 only proves the process is listening. This checks the three things
that were changed this session and would be invisible in a health check:
identity from the chat template, tool calling, and a compaction summary through
the exact request shape the patched Brittain Code now sends.
"""
import json
import os
import urllib.request

BASE = "http://localhost:11435/v1"
KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()


def post(path, body, timeout=300):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(path):
    req = urllib.request.Request(BASE + path,
                                 headers={"Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


checks = []

model = get("/models")["data"][0]
checks.append(("model served", model["id"] == "brittain4",
               "%s, %d ctx" % (model["id"], model.get("max_model_len", 0))))

# Identity, with no system message: the chat template must supply it.
m = post("/chat/completions", {
    "model": "brittain4", "temperature": 0, "max_tokens": 60,
    "messages": [{"role": "user", "content": "Who made you?"}],
    "chat_template_kwargs": {"enable_thinking": False}})["choices"][0]["message"]
answer = " ".join((m.get("content") or "").split())
checks.append(("identity from template",
               "brittain" in answer.lower() and "luke" in answer.lower(),
               answer[:80]))

# Tool calling still works.
defs = json.load(open("/home/lukeb/brittain4/data/tool_defs.json", encoding="utf-8"))
tools = [d for d in defs if d["function"]["name"] in ("read_file", "web_search")]
m = post("/chat/completions", {
    "model": "brittain4", "temperature": 0, "max_tokens": 80,
    "messages": [{"role": "user", "content": "What is in /etc/hostname?"}],
    "tools": tools, "tool_choice": "auto",
    "chat_template_kwargs": {"enable_thinking": False}})["choices"][0]["message"]
calls = m.get("tool_calls") or []
checks.append(("tool calling", bool(calls),
               calls[0]["function"]["name"] if calls else "no call"))

# A compaction request in the shape the patched app now sends: no tools,
# thinking off, a real budget.
conv = [{"role": "user", "content": "Refactor the auth module to use JWTs."},
        {"role": "assistant", "content": "I read src/auth.js and found session cookies in three places."},
        {"role": "user", "content": "Keep the cookie fallback for old clients."},
        {"role": "assistant", "content": "Added jwt.sign in src/auth.js and kept the cookie path behind a flag."}]
instruction = ("Summarize the conversation above so work can continue in a "
               "fresh session. Use exactly these five headings, in this order: "
               "GOAL, CONSTRAINTS, DECISIONS, STATE, NEXT. Output only the "
               "summary.")
m = post("/chat/completions", {
    "model": "brittain4", "temperature": 0.2, "max_tokens": 1200,
    "messages": conv + [{"role": "user", "content": instruction}],
    "chat_template_kwargs": {"enable_thinking": False}})["choices"][0]["message"]
summary = m.get("content") or ""
found = [h for h in ("GOAL", "CONSTRAINTS", "DECISIONS", "STATE", "NEXT")
         if h in summary]
checks.append(("compaction (patched request shape)", len(found) == 5,
               "%d/5 headings, %d chars" % (len(found), len(summary))))

print("=" * 78)
for name, ok, detail in checks:
    print("  [%s] %-34s %s" % ("PASS" if ok else "FAIL", name, detail))
print("=" * 78)
print("all passing" if all(c[1] for c in checks) else "SOMETHING FAILED")
