# -*- coding: utf-8 -*-
"""Scan the chat corpus for credentials before any of it becomes weights.

One chat is titled "AISD Portal Credentials and Google Classroom Access". These
are real exports of real sessions, and anything in them is a candidate for
memorisation -- a fine-tuned model can and does emit training data verbatim. A
password baked into an adapter cannot be removed without retraining, and if the
weights are ever released it goes with them.

Reports locations and match types only. It deliberately does NOT print the
matched secrets: this output goes into a transcript.
"""
import glob
import json
import os
import re
from collections import Counter

ROOT = "/home/lukeb/brittain4/data/chats"

PATTERNS = [
    ("private key",      re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("openai-style key",  re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("github token",      re.compile(r"\b(ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}")),
    ("aws access key",    re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack token",       re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("google api key",    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("bearer token",      re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}")),
    ("password= assign",  re.compile(r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S{4,}")),
    ("secret= assign",    re.compile(r"(?i)\b(api[_-]?key|secret|token)\s*[:=]\s*\S{8,}")),
    ("credential words",  re.compile(r"(?i)\b(my (password|login) is|username\s*[:=])")),
]

hits = Counter()
by_chat = {}

for path in glob.glob(os.path.join(ROOT, "**", "*.json"), recursive=True):
    if os.path.basename(path) == "index.json":
        continue
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
        title = (json.loads(text).get("title") or "")[:60]
    except Exception:
        continue
    found = [name for name, rx in PATTERNS if rx.search(text)]
    if found:
        for name in found:
            hits[name] += 1
        by_chat.setdefault(os.path.basename(path), (title, set()))[1].update(found)

print("scanned %d chat files\n" % len(glob.glob(os.path.join(ROOT, "**", "*.json"), recursive=True)))
print("=" * 74)
print("match types across the corpus")
print("=" * 74)
if not hits:
    print("  no matches")
for name, count in hits.most_common():
    print("  %-18s %d file(s)" % (name, count))

print()
print("=" * 74)
print("files with matches (types only -- values deliberately not printed)")
print("=" * 74)
for name, (title, kinds) in sorted(by_chat.items()):
    print("  %-34s %s" % (name, ", ".join(sorted(kinds))))
    if title:
        print("      title: %s" % title)
print()
print("NOTE: assignment-style patterns match ordinary code and config too, so")
print("these are candidates to review, not confirmed live credentials.")
