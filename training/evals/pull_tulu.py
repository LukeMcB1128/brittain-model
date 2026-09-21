# -*- coding: utf-8 -*-
"""Download Tulu 3 SFT and report what is in it, before sampling anything.

allenai/tulu-3-sft-mixture, odc-by (confirmed from the dataset card API, not
recalled). Six parquet shards, 1.41 GB, so it is taken whole rather than
streamed -- streaming and taking the first N would sample whatever subset the
mixture happens to order first, which is not a sample of the mixture.

This step only inspects. What proportion to take is a decision that needs the
token numbers, and those are measured in the next step rather than guessed.
"""
import collections
import json
import os
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "allenai/tulu-3-sft-mixture"
DEST = "/home/lukeb/brittain4/data/tulu"
os.makedirs(DEST, exist_ok=True)

paths = []
for i in range(6):
    name = "data/train-%05d-of-00006.parquet" % i
    sys.stdout.write("  downloading %s ... " % name.split("/")[-1])
    sys.stdout.flush()
    path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=DEST)
    paths.append(path)
    print("%.0f MB" % (os.path.getsize(path) / 1e6))

print("\nreading schema and composition...")
sources = collections.Counter()
turn_counts = collections.Counter()
chars = collections.Counter()
total_rows = 0
schema_printed = False

for path in paths:
    table = pq.read_table(path, columns=None)
    if not schema_printed:
        print("\ncolumns: %s" % table.schema.names)
        schema_printed = True
    cols = {name: table.column(name).to_pylist() for name in table.schema.names
            if name in ("source", "messages", "id")}
    n = table.num_rows
    total_rows += n
    for i in range(n):
        src = (cols.get("source") or [None] * n)[i] or "?"
        sources[src] += 1
        msgs = (cols.get("messages") or [None] * n)[i] or []
        turn_counts[min(len(msgs), 9)] += 1
        size = sum(len(str(m.get("content") or "")) for m in msgs)
        # Buckets, because the mean is dominated by a long tail.
        chars["<2k" if size < 2000 else "2-8k" if size < 8000 else "8-32k" if size < 32000 else ">32k"] += 1

print("\ntotal rows: %d" % total_rows)
print("\nsubsets by share:")
for name, count in sources.most_common(24):
    print("  %-46s %7d  %5.1f%%" % (name[:46], count, 100.0 * count / total_rows))

print("\nmessages per example: %s"
      % {k: v for k, v in sorted(turn_counts.items())})
print("size buckets        : %s" % dict(chars))
