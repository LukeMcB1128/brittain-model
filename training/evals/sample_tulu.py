# -*- coding: utf-8 -*-
"""Sample a general-instruction slice from Tulu 3, by skill rather than by share.

WHY NOT PROPORTIONAL
Tulu's own weighting is 43% mathematics (personahub_math 16%, numinamath 6.8%,
open_math_gsm8k 5.3%, personas-math-grade 5.3%, personahub_math_interm 2.1%).
Sampling proportionally would make nearly half of this model's general data
maths drills. The slice exists to stop general ability drifting while the
adapter learns tool-driving -- for that, breadth matters and Tulu's mixture
weighting does not.

WHAT IS EXCLUDED OUTRIGHT
  tulu_hard_coded_repeated_10  240 rows of identity answers -- "I am Tulu,
                               trained by Ai2". Training on those directly
                               fights the 161 identity examples that say
                               BRITTAIN-4, made by Luke Brittain. This is the
                               single most important exclusion in the file.

MULTILINGUAL IS A DELIBERATE, SEPARATE DIAL
  tulu_v3.9_aya_100k is 10.6% of the mixture and is non-English. Kept at a
  small share rather than proportionally: some breadth is useful, but at this
  sample size a proportional slice would be a noticeable fraction of everything
  the model sees, against a trajectory set that is entirely English.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import random

import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000, help="examples to sample")
ap.add_argument("--out", default="/home/lukeb/brittain4/data/general_sft.jsonl")
ap.add_argument("--max-chars", type=int, default=12000,
                help="skip examples longer than this; training windows at 4096 tokens")
args = ap.parse_args()

# Target shares by skill, not by Tulu's own proportions.
RECIPE = {
    "chat and instruction following": (0.34, [
        "tulu_v3.9_wildchat_100k", "no_robots_converted", "oasst1_converted",
        "flan_v2_converted", "personahub_ifdata_manual_seed_v3",
    ]),
    "code": (0.20, ["evol_codealpaca_heval", "personahub_code_v2"]),
    "maths": (0.20, [
        "personahub_math_v5_regen", "numinamath_tir_math", "open_math_2_gsm8k",
        "tulu-3-sft-personas-math-grade", "personahub_math_interm",
    ]),
    "safety and refusal": (0.16, [
        "wildjailbreak_decontam", "synthetic_finalresp_wi", "coconot_converted",
    ]),
    "science and tables": (0.05, ["sciriff_10k", "table_gpt_5k"]),
    "multilingual": (0.05, ["tulu_v3.9_aya_100k"]),
}
# Identity answers for a different assistant. Never sampled.
BANNED = ("tulu_hard_coded",)


def bucket_of(source):
    if any(b in source for b in BANNED):
        return None
    for name, (_, patterns) in RECIPE.items():
        if any(p in source for p in patterns):
            return name
    return None


def usable(messages):
    """Two-party, non-empty, ends on the assistant, and short enough to train."""
    if not messages or len(messages) < 2:
        return False
    if messages[-1].get("role") != "assistant":
        return False
    if any(not str(m.get("content") or "").strip() for m in messages):
        return False
    if any(m.get("role") not in ("user", "assistant", "system") for m in messages):
        return False
    return sum(len(str(m.get("content") or "")) for m in messages) <= args.max_chars


# Reservoir per bucket, so one pass over 939k rows needs no sorting and no
# second read, and the choice does not depend on shard order.
rng = random.Random(4)
wanted = {name: int(args.n * share) for name, (share, _) in RECIPE.items()}
pool = {name: [] for name in RECIPE}
seen = collections.Counter()
skipped = collections.Counter()

for path in sorted(glob.glob("/home/lukeb/brittain4/data/tulu/data/*.parquet")):
    table = pq.read_table(path, columns=["id", "messages", "source"])
    ids = table.column("id").to_pylist()
    msgs = table.column("messages").to_pylist()
    srcs = table.column("source").to_pylist()
    for i in range(table.num_rows):
        bucket = bucket_of(srcs[i] or "")
        if bucket is None:
            skipped["not in the recipe or banned"] += 1
            continue
        if not usable(msgs[i]):
            skipped["too long or malformed"] += 1
            continue
        seen[bucket] += 1
        reservoir = pool[bucket]
        if len(reservoir) < wanted[bucket]:
            reservoir.append((ids[i], srcs[i], msgs[i]))
        else:
            j = rng.randrange(seen[bucket])
            if j < wanted[bucket]:
                reservoir[j] = (ids[i], srcs[i], msgs[i])

rows = []
for bucket, items in pool.items():
    for ident, source, messages in items:
        rows.append({
            "kind": "general",
            "bucket": bucket,
            "source": source,
            "source_id": ident,
            "messages": [{"role": m["role"], "content": str(m["content"])} for m in messages],
        })
rng.shuffle(rows)

with open(args.out, "w", encoding="utf-8") as fh:
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

print("=" * 72)
print("%-32s %-8s %s" % ("bucket", "taken", "drawn from"))
print("=" * 72)
for bucket in RECIPE:
    print("%-32s %-8d %d candidates" % (bucket, len(pool[bucket]), seen[bucket]))
print("=" * 72)
print("total sampled : %d" % len(rows))
print("skipped       : %s" % dict(skipped))

# Prove the identity subset never made it in.
leaked = [r for r in rows if any(b in r["source"] for b in BANNED)]
print("\ntulu identity rows included: %d %s"
      % (len(leaked), "(must be 0)" if not leaked else "<-- LEAK"))
mentions = sum(1 for r in rows
               if "tulu" in json.dumps(r["messages"]).lower()
               or "ai2" in json.dumps(r["messages"]).lower())
print("examples mentioning tulu/ai2 anywhere: %d (review if non-zero)" % mentions)
print("\nwrote %s" % args.out)
