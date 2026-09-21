# -*- coding: utf-8 -*-
"""Is it the injection, the content, or just the length?

Reordering the guidance moved tool-calling 2/6 -> 3/6, so ordering was not the
cause. Three candidates remain, and they have different fixes:

  A. The template injects the text differently from how a caller's system
     message is rendered -> fix the Jinja.
  B. The specific wording talks the model out of tools -> reword.
  C. Any long system suffix after Qwen's tool block dilutes it -> keep the
     default prompt short, whatever it says.

Sending the SAME full text as an explicit caller system message separates A
from B and C: if that also collapses, the injection is faithful and the text is
at fault. A long block of filler prose that says nothing about tools then
separates C from B.
"""
import json
import os
import re
import urllib.request

BASE = "http://localhost:11435/v1/chat/completions"
KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()

# Pull the exact string the template uses, so this cannot drift from it.
tpl = open("/home/lukeb/brittain4/chat_template_brittain4.jinja",
           encoding="utf-8").read()
m = re.match(r'\{%-\s*set brittain_identity = "(.*?)"\s*%\}', tpl, re.S)
FULL = m.group(1).encode().decode("unicode_escape")
SHORT = FULL.split("\n\n")[0]          # identity paragraph only
STYLE_ONLY = "\n\n".join(FULL.split("\n\n")[:2])   # identity + style, no tools

# Same length as FULL, says nothing about tools or answering.
FILLER = ("You are a helpful assistant. " + (
    "Your responses should be considerate of the fact that users come from "
    "many different backgrounds and levels of familiarity with the subject "
    "matter at hand, and may be reading in a second language or under time "
    "pressure of one kind or another. Courtesy costs nothing. Where a user "
    "has been unclear it is usually better to make a reasonable assumption "
    "and say what you assumed. "))
FILLER = FILLER + " " * 0
while len(FILLER) < len(FULL):
    FILLER += ("Different people want different amounts of detail, and there "
               "is no single correct length for an answer. ")
FILLER = FILLER[:len(FULL)]

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


def run(label, sysmsg):
    hits = 0
    for task in TASKS:
        msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else [])
        msgs.append({"role": "user", "content": task})
        body = {"model": "brittain4", "temperature": 0, "max_tokens": 200,
                "messages": msgs, "tools": TOOLS, "tool_choice": "auto",
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(
            BASE, data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + KEY,
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            msg = json.load(r)["choices"][0]["message"]
        if msg.get("tool_calls"):
            hits += 1
    n = len(sysmsg) if sysmsg else 0
    print("  %-42s %d/%d   (system: %d chars)" % (label, hits, len(TASKS), n))
    return hits


print("=" * 72)
print("tool-calling rate by system message")
print("=" * 72)
run("template default (nothing sent)", None)
run("FULL text sent as caller system msg", FULL)
run("identity paragraph only", SHORT)
run("identity + style, no tool guidance", STYLE_ONLY)
run("filler of the same length", FILLER)
run("nothing at all is impossible here; neutral", "You are a helpful assistant.")
print("=" * 72)
