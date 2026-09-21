"""What did the ten new sessions actually produce?

Volume is the easy part. The things worth checking are the ones the earlier
harvest was missing and that only certain prompts can supply:
  * ask_user  -- the model correctly stopping to ask, of which there were 4
                 examples in the entire corpus
  * refusals  -- every trajectory so far ended in success
  * thinking off -- 87/6 skew before this batch
  * short sessions -- the corpus is dominated by long ones
"""
import collections
import glob
import json
import os
import sys

NEW_DIR, OLD_DIR = sys.argv[1], sys.argv[2]
old = {os.path.basename(p) for p in glob.glob(os.path.join(OLD_DIR, "**", "*.json"), recursive=True)}

REFUSAL = ("i can't", "i cannot", "i won't", "i will not", "i'm not able",
           "i am not able", "not appropriate", "i must decline", "refuse",
           "i'd rather not", "i would not recommend", "i'm unable")

rows = []
for p in sorted(glob.glob(os.path.join(NEW_DIR, "**", "*.json"), recursive=True)):
    name = os.path.basename(p)
    if name in old or name == "index.json":
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except ValueError:
        continue
    conv = d.get("conversation") or []
    calls = collections.Counter(
        (c.get("function") or {}).get("name")
        for m in conv for c in (m.get("tool_calls") or []))
    errs = sum(1 for m in conv if m.get("role") == "tool"
               and str(m.get("content") or "").lstrip().lower().startswith("error:"))
    refused = any(m.get("role") == "assistant"
                  and any(k in str(m.get("content") or "").lower() for k in REFUSAL)
                  for m in conv)
    rows.append({
        "name": name, "title": (d.get("title") or "")[:40],
        "mode": d.get("mode"), "think": d.get("think"),
        "model": (d.get("model") or "?"),
        "msgs": len(conv), "calls": sum(calls.values()), "errors": errs,
        "ask_user": calls.get("ask_user", 0), "refused": refused,
        "tools": calls,
    })

print("%-38s %-5s %-6s %5s %5s %4s %4s %s"
      % ("title", "mode", "think", "msgs", "calls", "err", "ask", "refused"))
for r in rows:
    print("%-38s %-5s %-6s %5d %5d %4d %4d %s"
          % (r["title"], r["mode"], r["think"], r["msgs"], r["calls"],
             r["errors"], r["ask_user"], "YES" if r["refused"] else ""))

print("\n--- totals across the batch ---")
print("sessions      : %d" % len(rows))
print("messages      : %d" % sum(r["msgs"] for r in rows))
print("tool calls    : %d" % sum(r["calls"] for r in rows))
print("error results : %d" % sum(r["errors"] for r in rows))
print("ask_user      : %d" % sum(r["ask_user"] for r in rows))
print("refusals      : %d sessions" % sum(r["refused"] for r in rows))
print("thinking off  : %d of %d" % (sum(1 for r in rows if r["think"] is False), len(rows)))
print("modes         : %s" % dict(collections.Counter(r["mode"] for r in rows)))

allt = collections.Counter()
for r in rows:
    allt.update(r["tools"])
print("\ntools used:")
for t, n in allt.most_common(25):
    print("   %-34s %3d" % (t, n))
for t in ("append_file", "pdf_fill_form"):
    print("   %-34s %s" % (t, allt.get(t, 0)))
