# -*- coding: utf-8 -*-
"""What throughput is the server ACTUALLY delivering, from its own logs?

The 17 tok/s figure came from a synthetic benchmark, and that benchmark has
already been wrong once in each direction. The first version sent identical
prompts, so vLLM's prefix cache stored the KV once and it claimed 2,505
responses/hour. The fix sent fully distinct prompts -- which removes prefix
reuse altogether, and real traffic is nothing like that: every turn re-sends
the same system prompt and the same tool schemas, so the shared prefix is
genuinely reusable and the server's own logs show hit rates above 70%.

So neither synthetic number describes the deployment. vLLM logs actual
generation throughput alongside the number of running requests, which is
measured on real traffic and needs no assumptions at all.
"""
import re
import sys
from collections import defaultdict

LOG = "/home/lukeb/brittain4/serve.log"

# Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput:
# 33.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 27.4%,
# Prefix cache hit rate: 73.1%
LINE = re.compile(
    r"Avg prompt throughput:\s*([\d.]+) tokens/s.*?"
    r"Avg generation throughput:\s*([\d.]+) tokens/s.*?"
    r"Running:\s*(\d+) reqs.*?"
    r"Waiting:\s*(\d+) reqs.*?"
    r"Prefix cache hit rate:\s*([\d.]+)%")

# Only the current run: the log is appended across restarts.
text = open(LOG, encoding="utf-8", errors="replace").read()
marker = "=== starting new server run ==="
runs = text.split(marker)
current = runs[-1]
print("analysing the most recent run (%d of %d runs in the log)\n"
      % (len(runs), len(runs)))

by_concurrency = defaultdict(list)
prefix_rates = []
prompt_tps = []

for line in current.splitlines():
    m = LINE.search(line)
    if not m:
        continue
    prompt_t, gen_t, running, waiting, prefix = m.groups()
    running = int(running)
    gen_t = float(gen_t)
    prefix_rates.append(float(prefix))
    if float(prompt_t) > 0:
        prompt_tps.append(float(prompt_t))
    # Idle samples say nothing about speed.
    if running > 0 and gen_t > 0:
        by_concurrency[running].append(gen_t)

if not by_concurrency:
    print("no generation samples in this run yet -- has any traffic hit it?")
    sys.exit(0)


def pct(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


print("=" * 78)
print("%-12s %-8s %-10s %-10s %-10s %s"
      % ("concurrency", "samples", "median", "p90", "max", "per-stream median"))
print("=" * 78)
for running in sorted(by_concurrency):
    values = by_concurrency[running]
    median = pct(values, 0.5)
    print("%-12s %-8d %-10.1f %-10.1f %-10.1f %.1f tok/s"
          % (running, len(values), median, pct(values, 0.9), max(values),
             median / running))
print("=" * 78)

allv = [v for values in by_concurrency.values() for v in values]
print("\naggregate generation throughput: median %.1f, p90 %.1f, max %.1f tok/s"
      % (pct(allv, 0.5), pct(allv, 0.9), max(allv)))
if prompt_tps:
    print("prefill throughput when active : median %.0f, max %.0f tokens/s"
          % (pct(prompt_tps, 0.5), max(prompt_tps)))
if prefix_rates:
    print("prefix cache hit rate          : median %.1f%%, max %.1f%%"
          % (pct(prefix_rates, 0.5), max(prefix_rates)))
print("""
Note on reading this: "Avg generation throughput" is the engine total across
all running requests, so per-stream speed is that divided by concurrency. A
single request at 50 tok/s and two at 25 each both log as 50.""")
