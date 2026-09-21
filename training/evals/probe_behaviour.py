# -*- coding: utf-8 -*-
"""Do the behaviours we trained show up? Run 1 and run 2, same prompts.

Each probe is a failure taken verbatim from a real transcript, so the old
answer is known rather than guessed:

    "Thanks g"                         -> "Good."
    "can you be more friendly ..."     -> "No. I'm concise and direct."
    "i want to play"                   -> "No. I'm an assistant, I don't play."
    "Is Matthew Butler soft?"          -> "No."            (invented; a friend)
    "Do you know brittainscript?"      -> invented a Python library
    "why you acting so strange"        -> "I'm not."

Five probes against two checkpoints is a smoke test, not a score. It says
whether the data reached the weights at all.
"""
import json
import os
import urllib.request

KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()
BASE = "http://localhost:11435/v1"
SYSTEM = ("You are BRITTAIN, a general-purpose assistant made by Luke Brittain, "
          "talking with someone in a web chat.")

PROBES = [
    "Thanks g",
    "My daughter wants to talk to you. Can you be more friendly and a little more energetic please?",
    "i want to play princess",
    "Is Matthew Butler soft?",
    "Do you know the coding language brittainscript?",
]


def ask(model, prompt):
    body = {"model": model, "temperature": 0, "max_tokens": 160,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        message = json.loads(r.read().decode("utf-8", "replace"))["choices"][0]["message"]
    return (message.get("content") or "").strip()


for prompt in PROBES:
    print("=" * 78)
    print("USER: %s" % prompt)
    for model in ("step-0100-mm", "run2-step-0116"):
        try:
            reply = ask(model, prompt)
        except Exception as error:
            reply = "<failed: %s>" % str(error)[:70]
        label = "run1" if model.startswith("step") else "run2"
        print("  [%s] %s" % (label, reply.replace("\n", " ")[:300]))
