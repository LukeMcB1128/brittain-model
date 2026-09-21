# -*- coding: utf-8 -*-
"""What do conversation messages actually look like? The compaction repro's
filter (role in user/assistant, content is a non-empty string, >=12 of them)
matched nothing, so the assumption is wrong somewhere."""
import glob
import json
import os
from collections import Counter

roles = Counter()
ctypes = Counter()
per_chat = []

for path in glob.glob("/home/lukeb/brittain4/data/chats/chats_v5/*.json"):
    if os.path.basename(path) == "index.json":
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    conv = data.get("conversation") or []
    usable = 0
    for m in conv:
        roles[m.get("role")] += 1
        ctypes[type(m.get("content")).__name__] += 1
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"].strip():
            usable += 1
    per_chat.append((usable, len(conv), os.path.basename(path), data.get("title")))

print("roles across chats_v5 : %s" % dict(roles))
print("content types         : %s" % dict(ctypes))
print("\ntop chats by usable user/assistant string messages:")
for usable, total, name, title in sorted(per_chat, reverse=True)[:8]:
    print("  usable=%3d of %3d  %-34s %s" % (usable, total, name, (title or "")[:44]))

# Show one assistant message that is not a plain string.
for path in glob.glob("/home/lukeb/brittain4/data/chats/chats_v5/*.json"):
    if os.path.basename(path) == "index.json":
        continue
    data = json.load(open(path, encoding="utf-8"))
    for m in data.get("conversation") or []:
        if not isinstance(m.get("content"), str):
            print("\nexample non-string content: role=%s keys=%s" % (m.get("role"), list(m)))
            print("  content: %s" % json.dumps(m.get("content"))[:300])
            raise SystemExit
