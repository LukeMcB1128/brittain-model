"""The tools with no training coverage, with their real schemas.

Half of these are the ones the spec flags as "defined but in no role set" --
present in code mode only because activeToolDefs() sends the unfiltered array.
Nobody reaches for them naturally, so they cannot be collected by using the app
harder; they have to be targeted deliberately.
"""
import collections
import json
import sys

UNCOVERED = [
    "append_file", "browser_type", "create_directory", "create_git_branch",
    "finalize_research", "find_references", "get_environment_variables",
    "list_processes", "local_http_request", "move_file", "pdf_fill_form",
    "project_outline", "record_observation", "revert_to_last_commit",
    "search_local_docs", "stop_process",
]
# From the spec: reachable only because code mode sends the whole array.
NO_ROLE_SET = {
    "get_environment_variables", "start_process", "process_status", "stop_process",
    "local_http_request", "create_git_branch", "revert_to_last_commit",
    "get_git_graph", "list_processes", "initiate_research_session",
    "record_observation", "finalize_research",
}

defs = {d["function"]["name"]: d["function"]
        for d in json.load(open(sys.argv[1], encoding="utf-8"))}

groups = collections.OrderedDict([
    ("filesystem writes", ["append_file", "create_directory", "move_file"]),
    ("code navigation", ["find_references", "project_outline", "search_local_docs"]),
    ("git", ["create_git_branch", "revert_to_last_commit"]),
    ("processes & environment",
     ["list_processes", "stop_process", "get_environment_variables", "local_http_request"]),
    ("research log", ["record_observation", "finalize_research"]),
    ("other", ["browser_type", "pdf_fill_form"]),
])

for group, names in groups.items():
    print("=== %s ===" % group)
    for n in names:
        fn = defs.get(n)
        if not fn:
            print("  %-28s (not in tool_defs)" % n)
            continue
        params = fn.get("parameters") or {}
        req = params.get("required") or []
        opt = [k for k in (params.get("properties") or {}) if k not in req]
        flag = "  [no role set]" if n in NO_ROLE_SET else ""
        print("  %-28s%s" % (n, flag))
        print("      %s" % (fn.get("description") or "")[:150])
        print("      required: %s   optional: %s"
              % (req or "-", ", ".join(opt[:6]) or "-"))
    print()

print("of the %d uncovered, %d are in no role set"
      % (len(UNCOVERED), sum(1 for n in UNCOVERED if n in NO_ROLE_SET)))
