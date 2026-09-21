# -*- coding: utf-8 -*-
"""Split the harvest into train and held-out, by chat, by content hash.

WHY THIS HAS TO EXIST
Every one of the 14 chats in the current eval_driver.jsonl also appears in
trajectories.jsonl. Train on that and the driver eval scores the adapter on
conversations it memorised, so every checkpoint comparison against the phase-1
baseline is meaningless -- and picking a checkpoint on a meaningless number is
the failure this project has already had four times.

SPLIT BY CHAT, NOT BY EXAMPLE
One conversation yields many examples with growing context, so two examples
from the same chat share almost all their tokens. Splitting by example would
put near-copies on both sides and leak just as badly as no split at all.

HASH, NOT SHUFFLE
sha256 of the chat id decides the side. That is reproducible without storing a
seed, and stable as the corpus grows: adding chats never moves an existing one
across the line, so a later eval stays comparable to an earlier one.
"""
import argparse
import collections
import hashlib
import json

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", required=True)
ap.add_argument("--train", required=True)
ap.add_argument("--heldout", required=True)
ap.add_argument("--heldout-share", type=float, default=0.20,
                help="fraction of CHATS held out (default 0.20)")
ap.add_argument("--tools", help="tool_defs.json; the baked tools that must all "
                                "survive in the training side")
args = ap.parse_args()

rows = [json.loads(line) for line in open(args.src, encoding="utf-8")]
by_chat = collections.defaultdict(list)
for row in rows:
    by_chat[row.get("chat_id")].append(row)


def side(chat_id):
    """Byte-for-byte the rule build_driver_eval.py already uses.

    That script has always selected its eval items with this hash, so the eval
    was never the contaminated half -- what was missing was a TRAINING file
    that excluded the same chats. Reimplementing the split with a different
    formula would have produced a second, disagreeing partition and left the
    existing eval_driver.jsonl overlapping the new training set: the exact
    problem this is meant to remove.
    """
    digest = hashlib.sha256(str(chat_id).encode()).hexdigest()
    return "heldout" if int(digest[:8], 16) % 10_000 < args.heldout_share * 10_000 else "train"


assignment = {chat_id: side(chat_id) for chat_id in by_chat}


def baked_calls(examples, baked):
    names = set()
    for row in examples:
        for call in (row.get("target") or {}).get("tool_calls") or []:
            name = (call.get("function") or {}).get("name") or call.get("name")
            if name in baked:
                names.add(name)
    return names


# The point of the adapter is that all 55 tools are baked in. A pure hash split
# sends 12-21 of them entirely to the held-out side, so the model would never
# be trained on them at all -- a worse failure than a thin eval. Any chat that
# is the ONLY source of a baked tool is pulled back to training, and the tool is
# then simply untestable held-out, which is a known limit rather than a hole in
# the model.
rescued = []
if args.tools:
    baked = {d["function"]["name"] for d in json.load(open(args.tools, encoding="utf-8"))}
    sources = collections.defaultdict(set)
    for chat_id, examples in by_chat.items():
        for name in baked_calls(examples, baked):
            sources[name].add(chat_id)
    for name, chats in sorted(sources.items()):
        if all(assignment[c] == "heldout" for c in chats):
            # Keep the richest of them on the training side.
            keep = max(chats, key=lambda c: len(by_chat[c]))
            assignment[keep] = "train"
            rescued.append((name, keep))

train, heldout = [], []
for chat_id, examples in by_chat.items():
    (heldout if assignment[chat_id] == "heldout" else train).extend(examples)


def tools_in(examples):
    names = collections.Counter()
    for row in examples:
        for call in (row.get("target") or {}).get("tool_calls") or []:
            name = (call.get("function") or {}).get("name") or call.get("name")
            if name:
                names[name] += 1
    return names


with open(args.train, "w", encoding="utf-8") as fh:
    for row in train:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
with open(args.heldout, "w", encoding="utf-8") as fh:
    for row in heldout:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

train_chats = {c for c, s in assignment.items() if s == "train"}
held_chats = {c for c, s in assignment.items() if s == "heldout"}
train_tools, held_tools = tools_in(train), tools_in(heldout)

print("=" * 70)
print("%-12s %-8s %-10s %s" % ("", "chats", "examples", "distinct tools called"))
print("=" * 70)
print("%-12s %-8d %-10d %d" % ("train", len(train_chats), len(train), len(train_tools)))
print("%-12s %-8d %-10d %d" % ("held out", len(held_chats), len(heldout), len(held_tools)))
print("=" * 70)
print("held-out share: %.0f%% of chats, %.0f%% of examples"
      % (100.0 * len(held_chats) / len(by_chat),
         100.0 * len(heldout) / max(1, len(rows))))

overlap = train_chats & held_chats
print("\nchat overlap between the two sides: %d  %s"
      % (len(overlap), "(must be 0)" if not overlap else "<-- LEAK"))

# A held-out set that never exercises a tool cannot detect a regression in it.
missing = set(train_tools) - set(held_tools)
print("tools trained but never exercised held-out: %d" % len(missing))
if missing:
    print("  %s" % ", ".join(sorted(missing)[:18]))

if args.tools:
    baked = {d["function"]["name"] for d in json.load(open(args.tools, encoding="utf-8"))}
    trained_baked = set(train_tools) & baked
    print("\nbaked tools (%d total): %d trained, %d MISSING FROM TRAINING"
          % (len(baked), len(trained_baked), len(baked - trained_baked)))
    if baked - trained_baked:
        print("  %s" % ", ".join(sorted(baked - trained_baked)))
    print("baked tools also exercised held-out: %d" % len(set(held_tools) & baked))
    if rescued:
        print("\nchats pulled back to training to keep a baked tool covered: %d"
              % len({chat for _, chat in rescued}))
        for name, chat in rescued[:10]:
            print("  %-24s only in %s" % (name, chat))
print("\nwrote %s and %s" % (args.train, args.heldout))
