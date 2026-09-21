# -*- coding: utf-8 -*-
"""Which paragraph turns live-data questions into refusals?

Neutral scores 4/10 on live-data search, the BRITTAIN-4 default 0/10, and the
observed failure text was "I cannot provide today's exchange rate because I do
not have real-time access". That is a refusal, not an oversight, and the
suspicion is the identity paragraph: "do not discuss your architecture or
training data" generalising from a narrow topic ban into a broad "I lack
access" stance that then covers the open web.

Each paragraph is tested alone and in combination against the neutral baseline,
so the fix lands on the paragraph responsible rather than on whichever one is
easiest to reword.
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
# Same identity, but the topic ban is scoped so it cannot read as a general
# statement about what the model can reach.
IDENT_SCOPED = ("You are BRITTAIN-4, a general-purpose assistant made by Luke "
                "Brittain. That is your name and origin; answer questions "
                "about yourself on that basis. Your own architecture and "
                "training data are the one topic you decline; everything else "
                "you engage with normally.")
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

LIVE = [
    "Look up today's exchange rate for USD to EUR.",
    "What's the current price of Bitcoin?",
    "What's the weather in London right now?",
    "What's the latest stable version of Python?",
    "Who is the current CEO of Boeing?",
]

defs = json.load(open("/home/lukeb/brittain4/data/tool_defs.json",
                      encoding="utf-8"))
by_name = {d["function"]["name"]: d for d in defs}
TOOLS = [by_name[n] for n in
         ("read_file", "run_command", "web_search", "edit_file", "list_files")
         if n in by_name]


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
    ("neutral (baseline)", "You are a helpful assistant."),
    ("identity only", IDENT),
    ("identity SCOPED only", IDENT_SCOPED),
    ("tools para only", TOOLS_P),
    ("style para only", STYLE_P),
    ("full current", IDENT + "\n\n" + TOOLS_P + "\n\n" + STYLE_P),
    ("full, scoped identity", IDENT_SCOPED + "\n\n" + TOOLS_P + "\n\n" + STYLE_P),
]

print("=" * 80)
for label, sysmsg in CASES:
    hits = 0
    refusals = 0
    for t in LIVE:
        for _ in range(2):
            name, content = ask(sysmsg, t)
            hits += name == "web_search"
            low = content.lower()
            if any(p in low for p in ("i cannot", "i don't have real-time",
                                      "i do not have real-time",
                                      "no real-time", "don't have access",
                                      "do not have access")):
                refusals += 1
    print("  %-26s search %2d/10   'no access' replies %2d/10" % (label, hits, refusals))
print("=" * 80)
