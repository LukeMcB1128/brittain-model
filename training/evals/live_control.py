# -*- coding: utf-8 -*-
"""Is live-data search broken by the default prompt, or broken generally?

Every wording variant scored 0/5 on live-data questions, which means the
wording is not the lever -- but that only matters if a plain prompt does
better. If neutral is also near zero, the prompt is not the cause and rewording
it further is chasing something that was never mine to fix.

Repeats each question twice. The earlier numbers came from n=3-5 per cell, and
at that size a one-item swing looks like a finding. Anything inside +/-1 here
should be treated as noise, not signal.
"""
import json
import os
import re
import urllib.request

BASE = "http://localhost:11435/v1/chat/completions"
KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()

tpl = open("/home/lukeb/brittain4/chat_template_brittain4.jinja",
           encoding="utf-8").read()
FULL = re.match(r'\{%-\s*set brittain_identity = "(.*?)"\s*%\}',
                tpl, re.S).group(1).encode().decode("unicode_escape")

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
FIVE = [by_name[n] for n in
        ("read_file", "run_command", "web_search", "edit_file", "list_files")
        if n in by_name]
JUST_SEARCH = [by_name["web_search"]] if "web_search" in by_name else []


def ask(sysmsg, task, tools):
    msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else [])
    msgs.append({"role": "user", "content": task})
    body = {"model": "brittain4", "temperature": 0, "max_tokens": 120,
            "messages": msgs, "tools": tools, "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        m = json.load(r)["choices"][0]["message"]
    tc = m.get("tool_calls") or []
    return tc[0]["function"]["name"] if tc else None


CASES = [
    ("no system at all (bare)", None, FIVE),
    ("neutral: 'You are a helpful assistant.'", "You are a helpful assistant.", FIVE),
    ("BRITTAIN-4 default (served)", FULL, FIVE),
    ("BRITTAIN-4 default, only web_search offered", FULL, JUST_SEARCH),
    ("neutral, only web_search offered", "You are a helpful assistant.", JUST_SEARCH),
]

print("=" * 80)
for label, sysmsg, tools in CASES:
    hits, detail = 0, []
    for t in LIVE:
        for _ in range(2):
            name = ask(sysmsg, t, tools)
            hits += name == "web_search"
        detail.append("%s:%s" % (t.split()[0], name or "-"))
    print("  %-44s %2d/10   %s" % (label, hits, " ".join(detail)))
print("=" * 80)
print("(n=10 per row: 5 questions x 2 repeats. +/-1 is noise.)")
