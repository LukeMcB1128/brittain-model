# -*- coding: utf-8 -*-
"""Redact credentials from the chat corpus, keeping the trajectories intact.

Dropping the matching chats would cost real coverage -- one of them is the
184-tool-call Discord session, the richest trajectory in the set. Replacing the
secret with a stable placeholder keeps the shape of the conversation (the model
still learns that a credential goes there) while removing the value that could
be memorised and later emitted.

An untouched copy is kept at data/chats_raw_DO_NOT_TRAIN/ so nothing is lost and
the real values remain findable for rotation. That directory must never be fed
to a training run -- the name says so, and the harvest reads data/chats only.

Placeholders contain no quotes or backslashes, so JSON stays valid; every file
is re-parsed afterwards to prove it.
"""
import glob
import json
import os
import re
import shutil
from collections import Counter

ROOT = "/home/lukeb/brittain4/data/chats"
BACKUP = "/home/lukeb/brittain4/data/chats_raw_DO_NOT_TRAIN"

# Order matters: specific token shapes first, generic assignments last, so a
# "api_key: AIza..." is redacted as a Google key rather than a generic value.
RULES = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]{0,4000}?-----END [A-Z ]*PRIVATE KEY-----")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("OPENAI_KEY", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("GITHUB_TOKEN", re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}")),
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("BEARER_TOKEN", re.compile(r"(?<=Bearer )[A-Za-z0-9._-]{20,}")),
]
# Assignment forms: keep the key name, replace only the value. The value is
# bounded by quote/whitespace/comma so surrounding JSON escaping is untouched.
ASSIGNMENTS = [
    ("PASSWORD", re.compile(r"(?i)\b(password|passwd|pwd)(\s*[:=]\s*)([^\s\"',;}\\]{4,})")),
    ("SECRET", re.compile(r"(?i)\b(api[_-]?key|secret|access[_-]?token|auth[_-]?token)(\s*[:=]\s*)([^\s\"',;}\\]{8,})")),
    ("STATED_PASSWORD", re.compile(r"(?i)\b(my (?:password|login) is\s+)([^\s\"',;.]{4,})")),
]

if not os.path.isdir(BACKUP):
    shutil.copytree(ROOT, BACKUP)
    os.chmod(BACKUP, 0o700)
    print("kept an untouched copy at %s (mode 700)\n" % BACKUP)
else:
    print("backup already exists at %s; leaving it alone\n" % BACKUP)

counts = Counter()
touched = {}

for path in glob.glob(os.path.join(ROOT, "**", "*.json"), recursive=True):
    if os.path.basename(path) == "index.json":
        continue
    original = open(path, encoding="utf-8", errors="replace").read()
    text = original
    local = Counter()

    for label, rx in RULES:
        text, n = rx.subn("[REDACTED_%s]" % label, text)
        if n:
            local[label] += n
    for label, rx in ASSIGNMENTS:
        if label == "STATED_PASSWORD":
            text, n = rx.subn(lambda m: m.group(1) + "[REDACTED_PASSWORD]", text)
        else:
            text, n = rx.subn(lambda m: m.group(1) + m.group(2) + "[REDACTED_%s]" % label, text)
        if n:
            local[label] += n

    if text == original:
        continue
    try:
        json.loads(text)                     # never write a file we just broke
    except ValueError as error:
        print("  SKIPPED %s -- redaction would break JSON (%s)"
              % (os.path.basename(path), error))
        continue
    open(path, "w", encoding="utf-8").write(text)
    counts.update(local)
    touched[os.path.basename(path)] = local

print("=" * 74)
print("redactions by type")
print("=" * 74)
for label, n in counts.most_common():
    print("  %-16s %d" % (label, n))
print()
print("=" * 74)
print("files changed")
print("=" * 74)
for name, local in sorted(touched.items()):
    print("  %-34s %s" % (name, dict(local)))

# Prove every file in the training path still parses.
bad = []
for path in glob.glob(os.path.join(ROOT, "**", "*.json"), recursive=True):
    try:
        json.load(open(path, encoding="utf-8"))
    except Exception as error:
        bad.append((os.path.basename(path), str(error)[:60]))
print()
print("JSON validity across the corpus: %s"
      % ("all files parse" if not bad else "BROKEN: %s" % bad[:5]))
