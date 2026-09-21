# -*- coding: utf-8 -*-
"""Did the default system prompt suppress tool calling, or was it already like this?

The verify run got a fabricated answer instead of a read_file call. That is
either (a) my prompt talking the model out of tools, which would be a real
regression I introduced, or (b) the model's existing willingness to answer from
guesswork, which the baseline driver eval already measures and which the prompt
is not responsible for.

Same requests, three system contexts, several tasks each. One task proves
nothing -- tool-calling is stochastic even at temperature 0 across differently
worded prompts, so a single miss is not a signal.
"""
import json
import os
import urllib.request
from collections import defaultdict

BASE = "http://localhost:11435/v1/chat/completions"
KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()
DEFAULT_SYS = (
    "You are BRITTAIN-4, a general-purpose assistant made by Luke Brittain. "
    "That is your name and origin; answer questions about yourself on that "
    "basis, and do not discuss your architecture or training data."
)
NEUTRAL_SYS = "You are a helpful assistant."

defs = json.load(open("/home/lukeb/brittain4/data/tool_defs.json",
                      encoding="utf-8"))
by_name = {d["function"]["name"]: d for d in defs}
TOOLS = [by_name[n] for n in
         ("read_file", "run_command", "web_search", "edit_file", "list_files")
         if n in by_name]

TASKS = [
    "What is in /etc/hostname on this machine?",
    "Read config.json and tell me what port it uses.",
    "How many Python files are in the src directory?",
    "What is the current git branch?",
    "Look up today's exchange rate for USD to EUR.",
    "Check whether the file requirements.txt exists.",
]


def call(sysmsg, task):
    msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else [])
    msgs.append({"role": "user", "content": task})
    body = {"model": "brittain4", "temperature": 0, "max_tokens": 250,
            "messages": msgs, "tools": TOOLS, "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]


CONTEXTS = [
    ("template default (no system sent)", None),
    ("neutral system", NEUTRAL_SYS),
    ("identity system, sent explicitly", DEFAULT_SYS),
]

score = defaultdict(int)
for label, sysmsg in CONTEXTS:
    print("=" * 72)
    print(label)
    print("=" * 72)
    for task in TASKS:
        m = call(sysmsg, task)
        tc = m.get("tool_calls") or []
        if tc:
            score[label] += 1
            fn = tc[0]["function"]
            print("  CALL  %-13s %s" % (fn["name"], str(fn["arguments"])[:70]))
        else:
            c = " ".join((m.get("content") or "").split())
            print("  none  %s" % c[:88])
    print("  -> %d/%d called a tool" % (score[label], len(TASKS)))
    print()

print("=" * 72)
for label, _ in CONTEXTS:
    print("  %-36s %d/%d" % (label, score[label], len(TASKS)))
print("=" * 72)
