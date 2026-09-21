# -*- coding: utf-8 -*-
"""Two residual defects, tested on more than the one probe that revealed each.

1. LIVE DATA. The served prompt stated a USD/EUR rate of 0.92 from memory
   instead of searching. "Anything current is a tool call" was evidently too
   abstract. Confidently-wrong live data is the most user-visible failure on
   this list, so it is worth one targeted attempt -- but fixing a single probe
   is overfitting, so this scores five different live-data questions.

2. LITERAL HEADER. "Lead with the conclusion" produced a reply beginning
   "**Conclusion:**". The instruction is about order, not about labelling, and
   the model took it as a format. Counted across several explanation tasks.

Sent as caller system messages: ab_length.py established that the template
injection renders identically to a caller-sent system message, so wording can
be compared this way without a server restart per variant.
"""
import json
import os
import re
import urllib.request

BASE = "http://localhost:11435/v1/chat/completions"
KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()

IDENT = ("You are BRITTAIN-4, a general-purpose assistant made by Luke "
         "Brittain. That is your name and origin; answer questions about "
         "yourself on that basis, and do not discuss your architecture or "
         "training data.")

TOOLS_NOW = (
    "When tools are available, use them. Anything about this machine, its "
    "files, its command output, or anything current is a tool call, not a "
    "recollection -- never state such a thing from memory. Report what you "
    "found rather than narrating what you are about to do, and if a tool "
    "fails, say what actually failed.")

TOOLS_NEW = (
    "When tools are available, use them. Anything about this machine, its "
    "files, or its command output is a tool call, not a recollection. So is "
    "anything that changes over time -- prices, rates, versions, weather, "
    "news, who currently holds a position: search for it rather than "
    "answering from memory, however confident the number feels. Report what "
    "you found rather than narrating what you are about to do, and if a tool "
    "fails, say what actually failed.")

STYLE_NOW = (
    "Once you have what you need, lead with the conclusion and put the "
    "reasoning after it, and say plainly what you are unsure of instead of "
    "hedging everything. Prose for explanation, code blocks for code, a table "
    "only when comparing several things on the same dimensions.")

STYLE_NEW = (
    "Once you have what you need, answer directly: the substance first, the "
    "supporting detail after it, with no restatement of the question and no "
    "labelled preamble. Say plainly what you are unsure of instead of hedging "
    "everything. Prose for explanation, code blocks for code, a table only "
    "when comparing several things on the same dimensions.")

LIVE = [
    "Look up today's exchange rate for USD to EUR.",
    "What's the current price of Bitcoin?",
    "What's the weather in London right now?",
    "What's the latest stable version of Python?",
    "Who is the current CEO of Boeing?",
]
EXPLAIN = [
    "Explain the difference between a list and a tuple in Python.",
    "Why is quicksort usually faster than mergesort in practice?",
    "What does HTTP status 404 mean?",
    "Explain what a race condition is.",
]
# Regression guards: these must not move.
NEEDS_TOOL = [
    "What is in /etc/hostname on this machine?",
    "What is the current git branch?",
    "Check whether the file requirements.txt exists.",
]
NO_TOOL = ["What's 2+2?", "Write a haiku about rain."]

defs = json.load(open("/home/lukeb/brittain4/data/tool_defs.json",
                      encoding="utf-8"))
by_name = {d["function"]["name"]: d for d in defs}
TOOLS = [by_name[n] for n in
         ("read_file", "run_command", "web_search", "edit_file", "list_files")
         if n in by_name]
HEADER = re.compile(r"^\s*(\*\*|##)?\s*(conclusion|answer|summary|tl;dr)\b",
                    re.I)


def ask(sysmsg, task):
    body = {"model": "brittain4", "temperature": 0, "max_tokens": 160,
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": task}],
            "tools": TOOLS, "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        m = json.load(r)["choices"][0]["message"]
    return (bool(m.get("tool_calls")), " ".join((m.get("content") or "").split()))


VARIANTS = [
    ("current (tools_now + style_now)", TOOLS_NOW, STYLE_NOW),
    ("tools_new + style_now", TOOLS_NEW, STYLE_NOW),
    ("tools_new + style_new", TOOLS_NEW, STYLE_NEW),
    ("tools_now + style_new", TOOLS_NOW, STYLE_NEW),
]

print("=" * 84)
print("%-34s %-9s %-9s %-9s %s" % ("variant", "live/5", "hdr/4", "tool/3", "restr/2"))
print("=" * 84)
for label, tp, sp in VARIANTS:
    sysmsg = IDENT + "\n\n" + tp + "\n\n" + sp
    live = sum(ask(sysmsg, t)[0] for t in LIVE)
    hdr = sum(not HEADER.match(ask(sysmsg, t)[1]) for t in EXPLAIN)
    tool = sum(ask(sysmsg, t)[0] for t in NEEDS_TOOL)
    restr = sum(not ask(sysmsg, t)[0] for t in NO_TOOL)
    print("%-34s %-9s %-9s %-9s %s"
          % (label, "%d/5" % live, "%d/4" % hdr, "%d/3" % tool,
             "%d/2" % restr))
print("=" * 84)
print("live = searched instead of answering from memory")
print("hdr  = did NOT open with a labelled 'Conclusion:' preamble")
