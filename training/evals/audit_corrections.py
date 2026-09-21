# -*- coding: utf-8 -*-
"""Does the mix ever show the model being corrected and updating?

Live chat, about a private individual the model cannot know anything about:

    "Is matthew butler soft"   -> "No."
    "hes tuff?"                -> "Yes."
    "why is he tuff?"          -> "He's a strong player, that's why."
    "what does he play"        -> "He plays basketball."
    "I thought he played football" -> "No, he plays basketball."

Two failures. It invented facts about someone it has no knowledge of, and then
it overruled the user about the user's own friend.

The second is the one with a suspected cause. The identity set's pressure group
trains holding an answer against pushback -- correctly, for the name and the
maker. If nothing in the corpus shows an answer CHANGING when the user knows
better, "do not yield" is the only pattern available, and it generalises to
every assertion.

This counts user turns that read as a correction, and what the target does
next: concede and revise, or restate.
"""
import json
import re

MIX = "/home/lukeb/brittain4/data/train_mix.jsonl"

CORRECTION = re.compile(
    r"^\s*(no[,.]? |that'?s (not|wrong|incorrect)|actually[,. ]|i thought "
    r"|isn'?t it |wasn'?t it |you'?re wrong|that'?s backwards|nope\b"
    r"|it'?s actually)", re.I)
CONCEDE = re.compile(
    r"(you'?re right|you are right|my mistake|i was wrong|apolog|sorry"
    r"|good catch|thanks for the correction|i had that wrong|corrected)", re.I)
RESTATE = re.compile(r"^\s*(no[,.]|actually[,.]|it is|i'?m sure|that'?s right)", re.I)

rows = [json.loads(l) for l in open(MIX, encoding="utf-8") if l.strip()]

found = concede = restate = neither = 0
examples = []
for row in rows:
    messages = row.get("messages") or []
    if not messages:
        continue
    last_user = None
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                last_user = content
    if not last_user or messages[-1].get("role") != "user":
        continue
    if not CORRECTION.match(last_user.strip()):
        continue
    found += 1
    reply = (row["target"].get("content") or "").strip()
    if CONCEDE.search(reply):
        concede += 1
        if len(examples) < 3:
            examples.append((row["kind"], last_user[:60], reply[:110]))
    elif RESTATE.match(reply):
        restate += 1
    else:
        neither += 1

print("user turns that read as a correction: %d" % found)
print("  target concedes / revises : %d" % concede)
print("  target restates its claim : %d" % restate)
print("  neither                   : %d" % neither)

for kind, user, reply in examples:
    print("\n  [%s] user: %s" % (kind, user))
    print("        reply: %s" % reply)
