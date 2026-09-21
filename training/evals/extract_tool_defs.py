"""Pull TOOL_DEFS out of the driver spec into machine-readable JSON.

    python3 extract_tool_defs.py --spec <spec html> --out tool_defs.json

This file is load-bearing three times over, so it is worth having as data
rather than prose:
  * the eval validates emitted calls against these exact schemas,
  * the training prompt sends them (until they are baked into weights),
  * the ~120-token name index that stays in the prompt is derived from it.

The spec serializes each definition from require('./tools.js').TOOL_DEFS
rather than transcribing the source, so what is embedded here is the exact
object placed in the request body -- which is the only version that matters.
"""
import argparse
import html
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

doc = open(args.spec, encoding="utf-8").read()

# Each tool is a <details> carrying its name, with the definition in the last
# <pre class="verbatim"> inside it.
blocks = re.findall(
    r'<details class="tool" data-name="([^"]+)"(.*?)</details>', doc, re.S)
print("tool blocks found: %d" % len(blocks))

defs, failed = [], []
for name, body in blocks:
    pres = re.findall(r'<pre class="verbatim">(.*?)</pre>', body, re.S)
    if not pres:
        failed.append((name, "no definition block"))
        continue
    # Strip any markup inside the block, then unescape entities: descriptions
    # contain things like &lt;name&gt;-filled.pdf that must survive intact.
    raw = html.unescape(re.sub(r"<[^>]+>", "", pres[-1]))
    try:
        obj = json.loads(raw)
    except ValueError as e:
        failed.append((name, str(e)[:80]))
        continue
    got = (obj.get("function") or {}).get("name")
    if got != name:
        failed.append((name, "name mismatch: block says %r" % got))
        continue
    defs.append(obj)

print("parsed cleanly    : %d" % len(defs))
for name, why in failed:
    print("   FAILED %-28s %s" % (name, why))

names = [d["function"]["name"] for d in defs]
assert len(names) == len(set(names)), "duplicate tool names"

json.dump(defs, open(args.out, "w", encoding="utf-8"), indent=2)
print("\nwrote %s" % args.out)

# The numbers the spec quotes, recomputed here so a bad parse is obvious.
serialized = json.dumps(defs, separators=(",", ":"))
print("serialized chars  : %d" % len(serialized))
print("app's own estimate: %d tokens  (len/4, main.js:113)" % round(len(serialized) / 4))
print("name index        : %d chars" % len(", ".join(sorted(names))))

required = sum(len((d["function"].get("parameters") or {}).get("required") or []) for d in defs)
print("required params   : %d across %d tools" % (required, len(defs)))
