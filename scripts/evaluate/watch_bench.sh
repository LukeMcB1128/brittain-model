#!/bin/sh
# Live view of an in-progress universal_bench run. Reads the results JSON the
# run flushes after each checkpoint; safe to start at any time.
FILE="${1:-benchmarks/results/universal_benchmark_results.json}"
while :; do
  clear
  printf 'watching %s  —  %s\n\n' "$FILE" "$(date '+%H:%M:%S')"
  /usr/local/bin/python3 - "$FILE" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print("no results yet"); raise SystemExit
try:
    rows = json.loads(p.read_text())
except json.JSONDecodeError:
    print("mid-write, retrying"); raise SystemExit
print(f"{'checkpoint':34} {'style':10} {'pySyn':>6} {'pyP@1':>6} {'jsP@1':>6} {'tsP@1':>6} {'jsonSch':>8}")
for r in rows:
    t = r.get("tracks", {})
    g = lambda k, m: f"{100*t[k][m]:.1f}%" if k in t and m in t[k] else "-"
    print(f"{r['checkpoint'][:32]:34} {r.get('prompt_style','?'):10} "
          f"{g('python','syntax_validity'):>6} {g('python','pass@1'):>6} "
          f"{g('javascript','pass@1'):>6} {g('typescript','pass@1'):>6} {g('json','schema_validity'):>8}")
print(f"\n{len(rows)} checkpoint(s) done")
PY
  sleep 20
done
