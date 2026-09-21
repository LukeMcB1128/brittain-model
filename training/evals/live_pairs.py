# -*- coding: utf-8 -*-
"""Which PAIR of paragraphs produces the refusal?

Singles: tools 10/10, identity 8/10, style 8/10. All three together: 0/10, with
four "I don't have real-time access" replies. So no paragraph is individually
at fault and the interaction has to be found, not guessed.

The standing suspicion is tools x style: "never state such a thing from memory"
plus "say plainly what you are unsure of" composes into "I am unsure and must
not answer from memory" -- which is a refusal, and is exactly the observed text.
Pairs distinguish that from the alternatives instead of assuming it.

Also scores the plain-answer tasks, because a variant that searches for
everything is not a fix.
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
TOOLS_P = ("When tools are available, use them. Anything about this machine, "
           "its files, its command output, or anything current is a tool "
           "call, not a recollection -- never state such a thing from memory. "
           "Report what you found rather than narrating what you are about to "
           "do, and if a tool fails, say what actually failed.")
STYLE_P = ("Once you have what you need, lead with the conclusion and put the "
           "reasoning after it, and say plainly what you are unsure of "
           "instead of hedging everything. Prose for explanation, code blocks "
           "for code, a table only when comparing several things on the same "
           "dimensions.")
# Style with the uncertainty clause removed -- that clause is the half suspected
# of composing into a refusal.
STYLE_NOUNC = ("Once you have what you need, lead with the conclusion and put "
               "the reasoning after it. Prose for explanation, code blocks "
               "for code, a table only when comparing several things on the "
               "same dimensions.")

LIVE = [
    "Look up today's exchange rate for USD to EUR.",
    "What's the current price of Bitcoin?",
    "What's the weather in London right now?",
    "What's the latest stable version of Python?",
    "Who is the current CEO of Boeing?",
]
NO_TOOL = ["What's 2+2?", "Write a haiku about rain.",
           "What does HTTP status 404 mean?"]

defs = json.load(open("/home/lukeb/brittain4/data/tool_defs.json",
                      encoding="utf-8"))
by_name = {d["function"]["name"]: d for d in defs}
TOOLS = [by_name[n] for n in
         ("read_file", "run_command", "web_search", "edit_file", "list_files")
         if n in by_name]
REFUSAL = ("i cannot", "i don't have real-time", "i do not have real-time",
           "no real-time", "don't have access", "do not have access")


def ask(sysmsg, task):
    body = {"model": "brittain4", "temperature": 0, "max_tokens": 120,
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
    tc = m.get("tool_calls") or []
    return (tc[0]["function"]["name"] if tc else None,
            " ".join((m.get("content") or "").split()))


CASES = [
    ("identity + tools", IDENT + "\n\n" + TOOLS_P),
    ("identity + style", IDENT + "\n\n" + STYLE_P),
    ("tools + style", TOOLS_P + "\n\n" + STYLE_P),
    ("all three (current)", IDENT + "\n\n" + TOOLS_P + "\n\n" + STYLE_P),
    ("all three, style w/o uncertainty clause",
     IDENT + "\n\n" + TOOLS_P + "\n\n" + STYLE_NOUNC),
    ("identity + tools + style, style FIRST",
     IDENT + "\n\n" + STYLE_P + "\n\n" + TOOLS_P),
]

print("=" * 84)
print("%-42s %-12s %-12s %s" % ("variant", "search/10", "refusals/10", "restraint/3"))
print("=" * 84)
for label, sysmsg in CASES:
    hits = refus = 0
    for t in LIVE:
        for _ in range(2):
            name, content = ask(sysmsg, t)
            hits += name == "web_search"
            if any(p in content.lower() for p in REFUSAL):
                refus += 1
    restr = sum(ask(sysmsg, t)[0] is None for t in NO_TOOL)
    print("%-42s %-12s %-12s %s"
          % (label, "%d/10" % hits, "%d/10" % refus, "%d/3" % restr))
print("=" * 84)
