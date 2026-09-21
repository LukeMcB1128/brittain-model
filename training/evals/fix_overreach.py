# -*- coding: utf-8 -*-
"""The tools paragraph over-applied: it refused to write a history essay.

Reported failure: "write an essay about the US government in the early days of
the country" -> the model said it needed tools and could not write from memory.
The clause "never state such a thing from memory" was meant to cover this
machine and live data. It reads as absolute, and the model applied it to
general knowledge.

The eval that passed this wording is the reason it shipped: the restraint set
was five SHORT items (2+2, a haiku, "what does 404 mean"). None of them looks
like a research task, so none of them triggered the overreach. The restraint
set below is rebuilt around long-form knowledge work -- essays, history,
explanation, summary from memory -- which is where the failure actually lives.

Three axes now, and a variant has to hold all three:
  MACHINE  files/commands/git      -> must call a tool
  LIVE     rates, weather, current -> must search
  KNOWLEDGE essays, history, code  -> must answer directly, no tool
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

# Shipped wording -- the one that refuses essays.
TOOLS_SHIPPED = (
    "When tools are available, use them. Anything about this machine, its "
    "files, its command output, or anything current is a tool call, not a "
    "recollection -- never state such a thing from memory. Report what you "
    "found rather than narrating what you are about to do, and if a tool "
    "fails, say what actually failed.")

# Same rule, bounded to its two intended domains, with knowledge carved out.
TOOLS_BOUNDED = (
    "You have broad knowledge and should use it: questions about history, "
    "science, language, code, or anything you can reason about, you answer "
    "directly. Reach for a tool in two cases -- when the question is about "
    "this machine (its files, its command output, its repository), and when "
    "the answer changes over time (prices, rates, weather, versions, current "
    "events). In those two cases check rather than recall. Report what you "
    "found rather than narrating what you are about to do, and if a tool "
    "fails, say what actually failed.")

# Shorter variant of the same idea, in case length is doing work.
TOOLS_SHORT = (
    "Answer from your own knowledge by default. Use a tool when the question "
    "is about this machine -- its files, its command output, its repository "
    "-- or when the answer changes over time, such as prices, weather, or "
    "current events; check those rather than recalling them. Report what you "
    "found rather than narrating what you are about to do.")

MACHINE = [
    "What is in /etc/hostname on this machine?",
    "What is the current git branch?",
    "Check whether the file requirements.txt exists.",
    "How many Python files are in the src directory?",
    "What's the last commit message in this repo?",
]
LIVE = [
    "Look up today's exchange rate for USD to EUR.",
    "What's the current price of Bitcoin?",
    "What's the weather in London right now?",
    "What's the latest stable version of Python?",
]
# The set that would have caught this. Long-form knowledge work.
KNOWLEDGE = [
    "Write an essay about the US government in the early days of the country.",
    "Write a few paragraphs on the causes of the French Revolution.",
    "Explain how photosynthesis works.",
    "Summarise the argument of Hobbes' Leviathan.",
    "Write a short essay comparing the Roman Republic and the Roman Empire.",
    "Explain the difference between a list and a tuple in Python.",
    "Write a haiku about rain.",
    "What's 2+2?",
]

defs = json.load(open("/home/lukeb/brittain4/data/tool_defs.json",
                      encoding="utf-8"))
by_name = {d["function"]["name"]: d for d in defs}
TOOLS = [by_name[n] for n in
         ("read_file", "run_command", "web_search", "edit_file", "list_files")
         if n in by_name]
# A refusal without a tool call is its own failure: the model neither answered
# nor acted, which is what was reported.
REFUSAL = re.compile(
    r"(i cannot|i can't|i am unable|i'm unable|need to use|i don't have "
    r"access|i do not have access|from memory|without using)", re.I)


def ask(sysmsg, task, max_tokens=200):
    body = {"model": "brittain4", "temperature": 0, "max_tokens": max_tokens,
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


VARIANTS = [
    ("shipped (refuses essays)", IDENT + "\n\n" + TOOLS_SHIPPED),
    ("bounded", IDENT + "\n\n" + TOOLS_BOUNDED),
    ("bounded, short", IDENT + "\n\n" + TOOLS_SHORT),
]

print("=" * 88)
print("%-26s %-10s %-10s %-12s %s"
      % ("variant", "machine", "live", "knowledge", "refusals"))
print("=" * 88)
results = {}
for label, sysmsg in VARIANTS:
    mach = sum(ask(sysmsg, t)[0] is not None for t in MACHINE)
    live = sum(ask(sysmsg, t)[0] == "web_search" for t in LIVE)
    kn = ref = 0
    detail = []
    for t in KNOWLEDGE:
        name, content = ask(sysmsg, t)
        answered = name is None and not REFUSAL.search(content[:200])
        kn += answered
        if REFUSAL.search(content[:200]) or name is not None:
            ref += 1
            detail.append(t.split()[0] + ":" + (name or "refused"))
    results[label] = detail
    print("%-26s %-10s %-10s %-12s %s"
          % (label, "%d/%d" % (mach, len(MACHINE)), "%d/%d" % (live, len(LIVE)),
             "%d/%d" % (kn, len(KNOWLEDGE)), ref))
print("=" * 88)
for label, detail in results.items():
    print("  %-26s knowledge failures: %s" % (label, detail or "none"))
print()
print("--- the reported prompt, shipped wording vs best fix ---")
for label, sysmsg in VARIANTS:
    name, content = ask(sysmsg,
                        "Write an essay about the US government in the early "
                        "days of the country.", 220)
    print("\n[%s] tool=%s" % (label, name))
    print("  %s" % content[:260])
