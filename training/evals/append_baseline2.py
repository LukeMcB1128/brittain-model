# -*- coding: utf-8 -*-
"""Record the re-harvest and the split in BASELINE.md."""

SECTION = """

## Training data: re-harvested and split (supersedes earlier counts)

The harvest on disk was written on Sep 7 from an earlier set of chats. It is
renamed `trajectories.jsonl.stale-DO-NOT-TRAIN` — it predates the credential
redaction, contains duplicate copies of the same conversations, and every chat
in the driver eval also appeared in it.

### The duplication was invisible in the totals

`data/chats/` holds five exports taken at different times and they are nested
supersets: **535 files for 124 distinct conversations**. Walking them all
harvested the same chat up to five times. A first re-run produced 7,028
examples from what is really ~124 conversations — roughly 4-5× duplication,
which is straight memorisation pressure on whichever chats happened to be
exported most often. The harvest now keeps one file per chat id, choosing the
longest copy, because a conversation that continued between exports is more
complete in the later one.

| | |
|---|---|
| chat files on disk | 535 |
| distinct conversations | 124 |
| dropped as duplicate exports | 411 |
| chats harvested | 98 |
| **examples** | **1,669** (1,286 call a tool, 383 text-only) |
| error results | 171 |
| baked tools with no coverage | 0 of 55 |

### The contamination, and why it was not where it looked

All 14 chats in the old `eval_driver.jsonl` also appeared in the training file.
`build_driver_eval.py` was never the problem — it has always selected eval items
by `sha256(chat_id) % 10_000 < val_frac * 10_000`. What was missing was a
*training* file that excluded those same chats.

`evals/split_trajectories.py` produces one, reusing that exact formula. A
different formula would have created a second, disagreeing partition and left
the existing eval overlapping the new training set.

| | chats | examples |
|---|---|---|
| train | 69 | 1,087 |
| held out | 29 | 582 |
| overlap | **0** | **0** |

### One conflict, resolved toward training

A pure hash split sends some tools entirely to the held-out side, so the model
would never be trained on them — worse than a thin eval. Any chat that is the
only source of a baked tool is pulled back to training. One was:
`find_symbol` appears in exactly one conversation, `1788204853574`, which the
old eval also used. It stays in training, the eval was rebuilt from the
held-out file without it, and **`find_symbol` is therefore untestable held-out**
— a recorded limit rather than a silent overlap.

**All 55 baked tools are in the training set. 30 of them are also exercised
held-out**, so the driver eval can detect a regression in those 30 and not the
rest. That ceiling is a property of the corpus: 27 tools appear in only one
conversation each and can never be on both sides at any ratio.

### Field drift caught on the way

`build_driver_eval.py` read `r["think"]`, which the harvest stopped emitting
when it switched to `think_requested` (what the app asked for) and `has_trace`
(what the model actually did) — the chat-level flag having turned out to lie.
It now carries both.

Verify any future change with `evals/verify_split.py`, which asserts the
training file is disjoint from every eval set rather than that the splitter ran.
"""

p = "/home/lukeb/brittain4/BASELINE.md"
existing = open(p, encoding="utf-8").read()
if "Training data: re-harvested and split" in existing:
    raise SystemExit("section already present")
open(p, "a", encoding="utf-8").write(SECTION)
print("appended %d chars; BASELINE.md is now %d lines"
      % (len(SECTION), len(open(p, encoding="utf-8").read().splitlines())))
