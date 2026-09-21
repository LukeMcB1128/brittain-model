"""Assess the harvested Brittain Code chats as training data.

The questions that decide phase 2:
  * how many assistant turns actually call tools (that is the trainable signal),
  * how many calls name tools that no longer exist (11 were removed in a
    consolidation commit, and training on them teaches calls the dispatcher
    rejects outright),
  * which of the live 55 have no coverage at all, since those need a
    deliberate generation pass,
  * the code/chat and thinking split, because the model must serve both.
"""
import collections
import glob
import json
import os
import re
import sys

CHATS = sys.argv[1]
ARTIFACT = sys.argv[2]

# The live surface, taken from the spec's own serialization of TOOL_DEFS.
html = open(ARTIFACT, encoding="utf-8").read()
LIVE = set(re.findall(r'<details class="tool" data-name="([^"]+)"', html))
# Named in the spec's stability section as present in history, absent now.
REMOVED = {
    "analyze_file_structure", "calculate_file_hash", "count_lines", "file_info",
    "find_files", "find_largest_files", "get_file_type", "list_directory",
    "pattern_search_deep", "replace_in_file", "search_in_file",
}
print("live tools in spec: %d" % len(LIVE))

files = [f for f in sorted(glob.glob(os.path.join(CHATS, "**", "*.json"), recursive=True))
         if os.path.basename(f) != "index.json"]
print("chat files: %d\n" % len(files))

modes, models, think = collections.Counter(), collections.Counter(), collections.Counter()
roles = collections.Counter()
tool_use = collections.Counter()
dead_use = collections.Counter()
turns_with_calls = total_calls = dead_calls = 0
convs = msg_total = 0
per_chat_calls = []
err_results = 0
tool_results = 0

for path in files:
    try:
        data = json.load(open(path, encoding="utf-8"))
    except ValueError:
        continue
    conv = data.get("conversation") or []
    if not conv:
        continue
    convs += 1
    msg_total += len(conv)
    modes[data.get("mode", "?")] += 1
    models[data.get("model", "?")] += 1
    think[bool(data.get("think"))] += 1

    n_calls = 0
    for m in conv:
        roles[m.get("role", "?")] += 1
        if m.get("role") == "tool":
            tool_results += 1
            c = m.get("content") or ""
            if isinstance(c, str) and re.match(r"\s*(\[MCP auto-approved\]\s*)?(MCP tool error:|Error:)", c, re.I):
                err_results += 1
        for call in m.get("tool_calls") or []:
            name = (call.get("function") or {}).get("name", "?")
            total_calls += 1
            n_calls += 1
            tool_use[name] += 1
            if name in REMOVED or (name not in LIVE and name not in REMOVED):
                dead_use[name] += 1
                dead_calls += 1
        if m.get("tool_calls"):
            turns_with_calls += 1
    per_chat_calls.append(n_calls)

print("conversations with content : %d" % convs)
print("messages total             : %d" % msg_total)
print("roles                      : %s" % dict(roles))
print("assistant turns with calls : %d" % turns_with_calls)
print("tool calls total           : %d" % total_calls)
print("tool result messages       : %d  (of which Error:-prefixed: %d)"
      % (tool_results, err_results))
print("chats with >=1 tool call   : %d" % sum(1 for n in per_chat_calls if n))
print()
print("mode  :", dict(modes))
print("think :", dict(think))
print("models:", dict(models.most_common(8)))
print()
print("dead / unknown calls: %d of %d (%.1f%%)"
      % (dead_calls, total_calls, 100 * dead_calls / max(1, total_calls)))
for name, n in dead_use.most_common():
    tag = "removed" if name in REMOVED else "UNKNOWN"
    print("   %-28s %4d   %s" % (name, n, tag))
print()
print("top tools used:")
for name, n in tool_use.most_common(20):
    print("   %-28s %4d" % (name, n))
print()
uncovered = sorted(LIVE - set(tool_use))
print("live tools with ZERO coverage: %d of %d" % (len(uncovered), len(LIVE)))
print("   " + ", ".join(uncovered))
