# -*- coding: utf-8 -*-
"""Find response guidance that does not cost tool calls.

Established: the style paragraph is the cause (identity alone 6/6, identity +
style 0/6), and the injection itself is faithful, so this is purely a wording
search and can be run entirely through caller-sent system messages.

SCORED BOTH WAYS ON PURPOSE. A prompt that calls a tool for everything is not a
fix, it is the opposite failure -- "what is 2+2" does not need run_command. So
each variant is scored on tasks that need a tool AND on tasks that do not, and
a variant only passes if it gets both right. Optimising the first number alone
is how you end up shipping a model that shells out to answer arithmetic.
"""
import json
import os
import urllib.request

BASE = "http://localhost:11435/v1/chat/completions"
KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()

IDENT = ("You are BRITTAIN-4, a general-purpose assistant made by Luke "
         "Brittain. That is your name and origin; answer questions about "
         "yourself on that basis, and do not discuss your architecture or "
         "training data.")

TOOLS_PARA = (
    "When tools are available, use them. Anything about this machine, its "
    "files, its command output, or anything current is a tool call, not a "
    "recollection -- never state such a thing from memory. Report what you "
    "found rather than narrating what you are about to do.")

# The original, known to score 0/6 on its own.
STYLE_ORIG = (
    "When you write an answer, put the conclusion first and the reasoning "
    "after it, and say plainly what you are unsure of instead of hedging "
    "everything. Prose for explanation, code blocks for code, a table only "
    "when comparing several things on the same dimensions. Do not open by "
    "restating the question.")

# Scoped to the moment of writing a final reply, not to whether to act now.
STYLE_SCOPED = (
    "Once you have what you need, lead with the conclusion and put the "
    "reasoning after it, and say plainly what you are unsure of instead of "
    "hedging everything. Prose for explanation, code blocks for code, a table "
    "only when comparing several things on the same dimensions.")

# Formatting only -- no instruction about when or whether to answer.
STYLE_FORMAT = (
    "Prose for explanation, code blocks for code, a table only when comparing "
    "several things on the same dimensions. Say plainly what you are unsure "
    "of instead of hedging everything.")

VARIANTS = [
    ("identity only (control, 6/6)", IDENT),
    ("identity + orig style (control, 0/6)", IDENT + "\n\n" + STYLE_ORIG),
    ("identity + tools", IDENT + "\n\n" + TOOLS_PARA),
    ("identity + tools + scoped style", IDENT + "\n\n" + TOOLS_PARA + "\n\n" + STYLE_SCOPED),
    ("identity + tools + format-only style", IDENT + "\n\n" + TOOLS_PARA + "\n\n" + STYLE_FORMAT),
    ("identity + format-only style (no tools para)", IDENT + "\n\n" + STYLE_FORMAT),
]

NEEDS_TOOL = [
    "What is in /etc/hostname on this machine?",
    "Read config.json and tell me what port it uses.",
    "How many Python files are in the src directory?",
    "What is the current git branch?",
    "Look up today's exchange rate for USD to EUR.",
    "Check whether the file requirements.txt exists.",
    "What's the last commit message in this repo?",
    "Is the test suite currently passing?",
]
# Answerable from knowledge. A tool call here is overreach, not diligence.
NO_TOOL = [
    "What's 2+2?",
    "Explain the difference between a list and a tuple in Python.",
    "Write a haiku about rain.",
    "What does HTTP status 404 mean?",
    "Summarise the plot of Hamlet in one sentence.",
]

defs = json.load(open("/home/lukeb/brittain4/data/tool_defs.json",
                      encoding="utf-8"))
by_name = {d["function"]["name"]: d for d in defs}
TOOLS = [by_name[n] for n in
         ("read_file", "run_command", "web_search", "edit_file", "list_files")
         if n in by_name]


def called(sysmsg, task):
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
        return bool(json.load(r)["choices"][0]["message"].get("tool_calls"))


print("=" * 78)
print("%-44s %-9s %-9s %s" % ("variant", "uses/8", "restr/5", "total"))
print("=" * 78)
best = []
for label, sysmsg in VARIANTS:
    uses = sum(called(sysmsg, t) for t in NEEDS_TOOL)
    restraint = sum(not called(sysmsg, t) for t in NO_TOOL)
    total = uses + restraint
    best.append((total, uses, restraint, label))
    print("%-44s %-9s %-9s %d/13" % (label, "%d/8" % uses,
                                     "%d/5" % restraint, total))
print("=" * 78)
best.sort(reverse=True)
print("best: %s  (%d/8 uses, %d/5 restraint)"
      % (best[0][3], best[0][1], best[0][2]))
