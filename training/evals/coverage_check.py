"""Which of the uncovered tools did the new sessions actually reach?

The Field Notes prompt was written to exercise sixteen tools with no training
coverage, eight of which are in no role set and would never appear from normal
use. This reports what landed, what did not, and what the new sessions add on
top of the existing harvest.
"""
import collections
import glob
import json
import os
import sys

NEW_DIR, OLD_DIR = sys.argv[1], sys.argv[2]

UNCOVERED = [
    "append_file", "browser_type", "create_directory", "create_git_branch",
    "finalize_research", "find_references", "get_environment_variables",
    "list_processes", "local_http_request", "move_file", "pdf_fill_form",
    "project_outline", "record_observation", "revert_to_last_commit",
    "search_local_docs", "stop_process",
]


def chats(d):
    out = {}
    for p in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
        n = os.path.basename(p)
        if n == "index.json":
            continue
        try:
            out[n] = json.load(open(p, encoding="utf-8"))
        except ValueError:
            pass
    return out


new, old = chats(NEW_DIR), chats(OLD_DIR)
fresh = {k: v for k, v in new.items() if k not in old}
print("new sessions: %d\n" % len(fresh))

all_new_calls = collections.Counter()
for name, d in sorted(fresh.items(),
                      key=lambda kv: -len(kv[1].get("conversation") or [])):
    conv = d.get("conversation") or []
    calls = collections.Counter(
        (c.get("function") or {}).get("name")
        for m in conv for c in (m.get("tool_calls") or []))
    all_new_calls.update(calls)
    errs = sum(1 for m in conv if m.get("role") == "tool"
               and str(m.get("content") or "").lstrip().lower().startswith("error:"))
    print("%-36s msgs=%-5d calls=%-4d errors=%-3d mode=%-6s think=%s"
          % (name[:36], len(conv), sum(calls.values()), errs,
             d.get("mode", "?"), d.get("think")))
    print("    model: %s" % (d.get("model") or "?"))
    hits = [t for t in UNCOVERED if t in calls]
    if hits:
        print("    covers: %s" % ", ".join("%s x%d" % (t, calls[t]) for t in hits))

print("\n=== coverage of the 16 targets ===")
covered = [t for t in UNCOVERED if all_new_calls.get(t)]
missing = [t for t in UNCOVERED if not all_new_calls.get(t)]
for t in UNCOVERED:
    n = all_new_calls.get(t, 0)
    print("  %-28s %s" % (t, ("%d call(s)" % n) if n else "STILL MISSING"))
print("\ncovered %d of %d" % (len(covered), len(UNCOVERED)))
if missing:
    print("still missing: %s" % ", ".join(missing))

print("\n=== all tools used in the new sessions ===")
for t, n in all_new_calls.most_common():
    print("  %-34s %3d" % (t, n))
print("\ntotal new tool calls: %d" % sum(all_new_calls.values()))
