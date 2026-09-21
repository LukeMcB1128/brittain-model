# -*- coding: utf-8 -*-
"""Confirm nothing credential-shaped survived redaction.

The re-scan still reported "password= assign" matches, which is expected: the
scanner's own pattern matches `password: [REDACTED_PASSWORD]`. This checks the
thing that actually matters -- assignments whose value is NOT a placeholder --
so a real leftover cannot hide behind a placeholder-shaped match.
"""
import glob
import os
import re

ROOT = "/home/lukeb/brittain4/data/chats"

KEY = r"(?i)\b(password|passwd|pwd|api[_-]?key|secret|access[_-]?token|auth[_-]?token)"
VALUE = r"([^\s\"',;}\\]{4,})"
RX = re.compile(KEY + r"\s*[:=]\s*(?!\[REDACTED)" + VALUE)

# Values that look like assignments but are not secrets: env lookups, template
# placeholders, type annotations, and the words people write in prose.
BENIGN = ("process.env", "os.environ", "$", "<", "{", "your", "null", "none",
          "true", "false", "string", "str", "bool", "int", "required",
          "optional", "***", "xxx", "...", "[", "@", "%")

left = {}
for path in glob.glob(os.path.join(ROOT, "**", "*.json"), recursive=True):
    if os.path.basename(path) == "index.json":
        continue
    text = open(path, encoding="utf-8", errors="replace").read()
    for match in RX.finditer(text):
        value = match.group(2)
        if value.lower().startswith(BENIGN):
            continue
        left.setdefault(os.path.basename(path), set()).add(
            "%s=%s..." % (match.group(1).lower(), value[:3]))

print("files with a real-looking credential value remaining: %d" % len(left))
for name, kinds in sorted(left.items())[:15]:
    print("  %-34s %s" % (name, sorted(kinds)[:4]))
if not left:
    print("  none -- every remaining match is a placeholder or a benign value")
