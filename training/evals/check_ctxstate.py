"""Did the app record the context window it was actually running with?

The raw harvest replays the whole stored conversation, producing contexts up to
195,746 tokens. The model never saw that: modelReadyMessages() bounds results,
num_ctx caps the window, and auto-compaction fires at 0.70 of it. Training on
the full replay would teach a distribution that cannot occur at serving time.

Each chat carries contextState and runtime dicts. If they record the window or
the compaction state, the harvest can reconstruct what the model actually saw
instead of guessing a truncation rule.
"""
import collections
import glob
import json
import os
import sys

ctx_keys = collections.Counter()
rt_keys = collections.Counter()
samples = []

for path in glob.glob(os.path.join(sys.argv[1], "**", "*.json"), recursive=True):
    if os.path.basename(path) == "index.json":
        continue
    try:
        d = json.load(open(path, encoding="utf-8"))
    except ValueError:
        continue
    cs, rt = d.get("contextState") or {}, d.get("runtime") or {}
    ctx_keys.update(cs.keys())
    rt_keys.update(rt.keys())
    if len(samples) < 4 and cs:
        samples.append((os.path.basename(path), cs, rt, d.get("runMetrics") or {}))

    # A compaction leaves a marker on the message it replaced.
    for m in d.get("conversation") or []:
        if m.get("compactionRecord") or (m.get("meta") or "") == "compaction":
            ctx_keys["<<saw a compaction record>>"] += 1

print("contextState keys:", dict(ctx_keys))
print("runtime keys     :", dict(rt_keys))
print()
for name, cs, rt, rm in samples:
    print("--- %s" % name)
    print("   contextState:", json.dumps(cs))
    print("   runtime     :", json.dumps(rt)[:400])
    print("   runMetrics  :", json.dumps(rm)[:300])
    print()
