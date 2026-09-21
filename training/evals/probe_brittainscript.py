# -*- coding: utf-8 -*-
"""Run 2 says it knows BrittainScript. Does the code it writes actually run?

Claiming a capability it does not have would be a new failure replacing an old
one -- run 1 at least said honestly that it had never encountered the language.
So the generated program is executed, not eyeballed.
"""
import json
import os
import re
import subprocess
import tempfile
import urllib.request

KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()
BS = os.path.expanduser("~/venv/bin/bs")
FENCE = "`" * 3

TASKS = [
    ("count to five", "Write BrittainScript that prints the numbers 1 to 5. "
                      "Output only the code."),
    ("sum a list", "Write BrittainScript that adds up the numbers 1 through 10 "
                   "and prints the total. Output only the code."),
]


def ask(model, prompt):
    body = {"model": model, "temperature": 0, "max_tokens": 220,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        "http://localhost:11435/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as fh:
        return json.loads(fh.read().decode())["choices"][0]["message"].get("content") or ""


def strip_fence(text):
    if FENCE not in text:
        return text.strip()
    block = re.search(FENCE + r"[a-zA-Z]*\n(.*?)" + FENCE, text, re.S)
    return (block.group(1) if block else text).strip()


for label, prompt in TASKS:
    print("=" * 72)
    print("TASK: %s" % label)
    for model in ("step-0100-mm", "run2-step-0116"):
        code = strip_fence(ask(model, prompt))
        with tempfile.NamedTemporaryFile("w", suffix=".bs", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(code + "\n")
            path = fh.name
        try:
            done = subprocess.run([BS, path], capture_output=True, text=True,
                                  timeout=15)
            ran, out, err = done.returncode == 0, done.stdout, done.stderr
        except Exception as error:
            ran, out, err = False, "", str(error)[:60]
        finally:
            os.unlink(path)
        tag = "run1" if model.startswith("step") else "run2"
        print("\n  [%s] RUNS=%s" % (tag, ran))
        print("       code  : %s" % code.replace("\n", " | ")[:150])
        print("       stdout: %r" % (out or "").strip()[:60])
        if not ran:
            print("       error : %s" % (err or "").strip()[:110])
