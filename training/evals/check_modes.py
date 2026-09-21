"""Are 'brittain' and 'jarvis' the same dead mode under two names?

If it is a rename, every brittain chat should postdate every jarvis chat, and
the two should look structurally alike. If they interleave, they are two
different things and the guess is wrong.
"""
import collections
import glob
import json
import os
import sys

rows = []
for path in glob.glob(os.path.join(sys.argv[1], "**", "*.json"), recursive=True):
    if os.path.basename(path) == "index.json":
        continue
    try:
        d = json.load(open(path, encoding="utf-8"))
    except ValueError:
        continue
    mode = d.get("mode", "?")
    if mode not in ("jarvis", "brittain", "?"):
        continue
    conv = d.get("conversation") or []
    tools = collections.Counter(
        (c.get("function") or {}).get("name")
        for m in conv for c in (m.get("tool_calls") or []))
    rows.append({
        "mode": mode,
        "ts": d.get("timestamp", ""),
        "title": (d.get("title") or "")[:44],
        "model": d.get("model", "?"),
        "msgs": len(conv),
        "calls": sum(tools.values()),
        "tools": dict(tools.most_common(4)),
        "keys": sorted(set(k for m in conv for k in m)),
    })

for r in sorted(rows, key=lambda r: r["ts"]):
    print("%-9s %-24s %-22s msgs=%-4d calls=%-3d %s"
          % (r["mode"], r["ts"][:19], r["model"][:22], r["msgs"], r["calls"], r["title"]))

print()
for mode in ("jarvis", "brittain", "?"):
    grp = [r for r in rows if r["mode"] == mode]
    if not grp:
        continue
    ts = sorted(r["ts"] for r in grp if r["ts"])
    print("%-9s n=%-3d  %s .. %s" % (mode, len(grp), ts[0][:10] if ts else "?", ts[-1][:10] if ts else "?"))
    keys = collections.Counter()
    for r in grp:
        keys.update(r["keys"])
    print("          message keys: %s" % sorted(keys))
    tools = collections.Counter()
    for r in grp:
        tools.update(r["tools"])
    print("          tools: %s" % dict(tools.most_common(6)))
