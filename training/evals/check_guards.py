"""Are the degraded-model guards and syntax rejections present in the harvest?

Reading tools.js surfaced failure surfaces the serialized schemas do not show:

  self-talk        "you are leaking your reasoning into the file"
  shrinkage        "this overwrite SHRANK the file from N to M chars"
  futility         "STOP: this is consecutive rewrite #N"
  syntax rejection "Write/Edit/Append rejected - syntax error"

Each carries a prescribed next action, and each appears in TOOL OUTPUT the
model reads. If they are in the corpus they are among the most valuable rows in
it. They also matter for accounting: the harvester counts a tool result as a
failure only when it starts with "Error:", and none of these do -- so any that
exist are currently being logged as successes.
"""
import collections
import glob
import json
import os
import re
import sys

PATTERNS = {
    "self-talk warning": re.compile(r"leaking your reasoning into the file", re.I),
    "shrinkage warning": re.compile(r"overwrite SHRANK the file", re.I),
    "futility breaker": re.compile(r"consecutive rewrite #", re.I),
    "syntax rejection": re.compile(r"(?:Write|Edit|Append|Batch edit) rejected", re.I),
    "protected path": re.compile(r"is protected \(matched", re.I),
    "path escape": re.compile(r"Path escapes the working directory", re.I),
    "two-strike block": re.compile(r"already failed twice", re.I),
    "tool unavailable": re.compile(r"Tool unavailable", re.I),
    "approval denied": re.compile(r"(?:Cancelled by user|parked|deferred|unattended)", re.I),
    "growth refusal": re.compile(r"refusing\.$", re.I | re.M),
    "compaction": re.compile(r"conversation was compacted", re.I),
    "untrusted web notice": re.compile(r"UNTRUSTED EXTERNAL WEB CONTENT", re.I),
}

hits = collections.Counter()
examples = {}
tool_msgs = 0
error_prefixed = 0

for p in glob.glob(os.path.join(sys.argv[1], "**", "*.json"), recursive=True):
    if os.path.basename(p) == "index.json":
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except ValueError:
        continue
    for m in d.get("conversation") or []:
        content = str(m.get("content") or "")
        if m.get("role") == "tool":
            tool_msgs += 1
            if re.match(r"\s*(\[MCP auto-approved\]\s*)?(MCP tool error:|Error:)", content, re.I):
                error_prefixed += 1
        # A guard warning is only real if it came back from the tool that
        # emits it. Matching any message body finds Brittain Code's OWN SOURCE
        # read into context -- the code that generates the warning, not the
        # warning firing. That inflated every count on the first pass.
        name = m.get("tool_name") or ""
        emitters = {
            "self-talk warning": {"write_file", "edit_file", "edit_files", "append_file"},
            "shrinkage warning": {"write_file", "edit_files"},
            "futility breaker": {"write_file"},
            "syntax rejection": {"write_file", "edit_file", "edit_files", "append_file"},
            "protected path": {"write_file", "edit_file", "edit_files", "append_file",
                               "delete_file", "move_file", "copy_file", "create_directory"},
            "path escape": None,          # any tool can raise it
            "growth refusal": {"edit_file"},
            "untrusted web notice": {"web_search", "web_fetch"},
        }
        for label, pat in PATTERNS.items():
            if m.get("role") != "tool":
                continue
            allowed = emitters.get(label, None)
            if allowed is not None and name not in allowed:
                continue
            if pat.search(content):
                hits[label] += 1
                if label not in examples:
                    snippet = pat.search(content)
                    start = max(0, snippet.start() - 60)
                    examples[label] = (os.path.basename(p),
                                       content[start:snippet.end() + 120].replace("\n", " "))

print("tool result messages     : %d" % tool_msgs)
print("prefixed 'Error:'        : %d  (what the harvester counts as failure)" % error_prefixed)
print()
print("failure surfaces found in the corpus:")
for label in PATTERNS:
    n = hits.get(label, 0)
    mark = "" if n else "   <- absent"
    print("   %-24s %3d%s" % (label, n, mark))

print("\nfirst sighting of each:")
for label, (fname, snippet) in examples.items():
    print("\n  [%s]  %s" % (label, fname[:34]))
    print("     ...%s..." % snippet[:190])
