"""Find where a saved conversation keeps its turns.

The files are 40 KB with no top-level "messages" key, so the transcript lives
somewhere else. Print every key with the size of its serialized value and
recurse into whichever one is carrying the weight.
"""
import glob
import json
import os

d = os.path.join(os.environ["APPDATA"], "Brittain Code", "chats")

for path in sorted(glob.glob(os.path.join(d, "*.json"))):
    name = os.path.basename(path)
    if name == "index.json":
        continue
    data = json.load(open(path, encoding="utf-8"))
    print("\n=== %s (%d bytes on disk) ===" % (name, os.path.getsize(path)))
    for k, v in data.items():
        blob = json.dumps(v)
        kind = type(v).__name__
        extra = " len=%d" % len(v) if isinstance(v, (list, dict)) else ""
        print("  %-18s %-6s %8d bytes%s" % (k, kind, len(blob), extra))

    # Recurse into the largest container.
    big = max(data.items(), key=lambda kv: len(json.dumps(kv[1])))
    print("  -> largest key: %r" % big[0])
    val = big[1]
    if isinstance(val, list) and val:
        print("     list of %d; first element keys: %s"
              % (len(val), list(val[0]) if isinstance(val[0], dict) else type(val[0]).__name__))
        print("     first element:", json.dumps(val[0])[:500])
        roles = {}
        for m in val:
            if isinstance(m, dict):
                roles[m.get("role", "?")] = roles.get(m.get("role", "?"), 0) + 1
        print("     roles:", roles)
        has_calls = [m for m in val if isinstance(m, dict) and m.get("tool_calls")]
        print("     turns with tool_calls:", len(has_calls))
        if has_calls:
            print("     example call:", json.dumps(has_calls[0]["tool_calls"])[:400])
    elif isinstance(val, dict):
        print("     dict keys:", list(val)[:20])
        print("     sample:", json.dumps(val)[:400])
