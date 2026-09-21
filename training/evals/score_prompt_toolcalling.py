# -*- coding: utf-8 -*-
"""Score the served default on the same 13 tasks the wording was chosen on.

The sweep picked this wording by sending it as a caller system message. That is
not the path that ships -- the template injects it, and the shipped path is the
only number worth quoting. Measure on the stack actually served.

Also reports which specific tasks miss, because "5/6" with no detail is how a
real regression hides behind a plausible-looking score.
"""
import json
import os
import urllib.request

BASE = "http://localhost:11435/v1/chat/completions"
KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()

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
# The long-form entries are here because the first version of this set was all
# short items, and a prompt passed it 13/13 while refusing to write a history
# essay -- it called web_search instead, on the grounds that it could not write
# from memory. Short questions never triggered that; "write an essay" did.
# Keep the long-form knowledge tasks in this set.
NO_TOOL = [
    "What's 2+2?",
    "Explain the difference between a list and a tuple in Python.",
    "Write a haiku about rain.",
    "What does HTTP status 404 mean?",
    "Summarise the plot of Hamlet in one sentence.",
    "Write an essay about the US government in the early days of the country.",
    "Write a few paragraphs on the causes of the French Revolution.",
    "Summarise the argument of Hobbes' Leviathan.",
    "Explain how photosynthesis works.",
]

defs = json.load(open("/home/lukeb/brittain4/data/tool_defs.json",
                      encoding="utf-8"))
by_name = {d["function"]["name"]: d for d in defs}
TOOLS = [by_name[n] for n in
         ("read_file", "run_command", "web_search", "edit_file", "list_files")
         if n in by_name]


def ask(task):
    body = {"model": "brittain4", "temperature": 0, "max_tokens": 160,
            "messages": [{"role": "user", "content": task}],   # no system: the point
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


print("=" * 78)
print("SERVED PATH (template default, no system message sent)")
print("=" * 78)
uses = 0
for t in NEEDS_TOOL:
    name, content = ask(t)
    uses += bool(name)
    print("  %-5s %-52s %s" % ("CALL" if name else "MISS", t[:52],
                               name or content[:60]))
print("  -> uses a tool on %d/%d tasks that need one" % (uses, len(NEEDS_TOOL)))
print()
restraint = 0
for t in NO_TOOL:
    name, content = ask(t)
    restraint += not name
    print("  %-5s %-52s %s" % ("ok" if not name else "OVER", t[:52],
                               name or content[:60]))
print("  -> answers directly on %d/%d that need no tool"
      % (restraint, len(NO_TOOL)))
print("=" * 78)
print("TOTAL %d/%d   (same wording scored 13/13 when sent as a caller message)"
      % (uses + restraint, len(NEEDS_TOOL) + len(NO_TOOL)))
print("=" * 78)
