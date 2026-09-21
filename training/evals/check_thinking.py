"""Does the stored `think` flag match what the model actually did?

The flag records what the app REQUESTED. Over the API some models reason
regardless, so a chat can be marked think=false and still carry traces. If the
harvest trusts the flag, the thinking-off slice is contaminated.

Also checks whether traces live in their own field. If they do, thinking-off
training examples can be synthesised by dropping it, rather than needing runs
where the toggle is actually honoured.
"""
import collections
import glob
import json
import os
import sys

rows = collections.Counter()
by_provider = collections.defaultdict(collections.Counter)
examples = []

for p in glob.glob(os.path.join(sys.argv[1], "**", "*.json"), recursive=True):
    if os.path.basename(p) == "index.json":
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except ValueError:
        continue
    conv = d.get("conversation") or []
    if not conv:
        continue
    flag = bool(d.get("think"))
    # A trace in its own field, and a trace smuggled inline in content.
    traced = sum(1 for m in conv
                 if m.get("role") == "assistant" and (m.get("thinking") or "").strip())
    inline = sum(1 for m in conv
                 if m.get("role") == "assistant"
                 and "<think>" in str(m.get("content") or ""))
    provider = ((d.get("runtime") or {}).get("settings") or {}).get("provider", "?")
    key = (flag, traced > 0)
    rows[key] += 1
    by_provider[provider][key] += 1
    if flag is False and traced > 0 and len(examples) < 3:
        examples.append((os.path.basename(p), provider, d.get("model"), traced))
    if inline:
        rows["inline <think> in content"] += 1

print("(think flag, has separate trace) -> chats")
for k, n in sorted(rows.items(), key=lambda kv: str(kv[0])):
    print("   %-34s %d" % (str(k), n))

print("\nby provider:")
for prov, c in sorted(by_provider.items()):
    print("   %-10s %s" % (prov, dict(c)))

if examples:
    print("\nflag says OFF but traces present:")
    for name, prov, model, n in examples:
        print("   %-36s provider=%-8s model=%-22s traces=%d"
              % (name[:36], prov, str(model)[:22], n))
else:
    print("\nno chat marked think=false carries traces -- the flag is reliable so far")
