"""What does a saved Brittain Code conversation actually contain?

The question that decides phase 2: are tool CALLS and tool RESULTS both
persisted, or only the display text? Trajectory training needs the full
assistant turn -- content, tool_calls, and the tool messages that came back --
not the rendered transcript.
"""
import collections
import glob
import json
import os

d = os.path.join(os.environ["APPDATA"], "Brittain Code", "chats")
files = sorted(glob.glob(os.path.join(d, "*.json")))
print("files:", len(files))

for path in files:
    name = os.path.basename(path)
    data = json.load(open(path, encoding="utf-8"))
    if name == "index.json":
        print("\n--- index.json ---")
        print(json.dumps(data, indent=2)[:600])
        continue

    msgs = data.get("messages", data if isinstance(data, list) else [])
    print("\n--- %s ---" % name)
    print("top-level keys:", list(data)[:12] if isinstance(data, dict) else "(list)")
    print("messages:", len(msgs))
    roles = collections.Counter(m.get("role") for m in msgs)
    print("roles:", dict(roles))

    keys = collections.Counter()
    for m in msgs:
        keys.update(m.keys())
    print("message keys seen:", dict(keys))

    tool_calls = [m for m in msgs if m.get("tool_calls")]
    tool_results = [m for m in msgs if m.get("role") == "tool"]
    print("assistant turns with tool_calls:", len(tool_calls))
    print("tool result messages:", len(tool_results))

    if tool_calls:
        tc = tool_calls[0]["tool_calls"][0]
        print("example call:", json.dumps(tc)[:300])
    if tool_results:
        tr = tool_results[0]
        print("example result keys:", list(tr))
        print("example result:", json.dumps(tr.get("content"))[:220])
    names = collections.Counter(
        c.get("function", {}).get("name")
        for m in tool_calls for c in m["tool_calls"])
    if names:
        print("tools used:", dict(names))
