# -*- coding: utf-8 -*-
"""Assemble the four sets into one training mix.

ONE SHAPE FOR EVERY SOURCE
Each row becomes {messages: [...prompt...], target: {content, thinking,
tool_calls}}. The prompt is everything the model sees; the target is the single
assistant turn the loss is taken over. Rendering the target -- whether a
thinking trace is emitted, how a tool call is serialised -- is left to the
trainer, so that decision lives in exactly one place instead of being frozen
into the data.

TOOLS ARE STRIPPED FROM TRAJECTORY PROMPTS, AND THAT IS NOT OPTIONAL
The 55 schemas cost 8,082 tokens. Training windows at 4,096. A tools-present
example therefore cannot exist at this window -- the schemas alone are twice
the budget. This happens to be the plan anyway (bake the schemas so the app can
stop sending them), but it is worth being clear that the window forces it
rather than that it was freely chosen.

The consequence to watch: until the app stops sending schemas, serving sends a
prompt shape training never saw. The eval has to score BOTH ways -- tools sent
and not -- or a regression in the tools-present path goes unnoticed.

RATIO IS BY TRAINED TOKENS, NOT BY EXAMPLES
A trajectory step averages 12,132 tokens raw and a Tulu row 569. Counting rows
would call 2,000 general examples 62% of the mix; at the 4,096 window they are
26% of what the model actually sees. Tokens are what the loss is over.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scrub import residue, scrub  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/lukeb/brittain4/data")
ap.add_argument("--out", default="/home/lukeb/brittain4/data/train_mix.jsonl")
ap.add_argument("--window", type=int, default=4096)
ap.add_argument("--seed", type=int, default=4)
ap.add_argument("--general", type=int, default=800,
                help="Tulu rows to keep. Run 1 used 2,000, which was 71.5%% of "
                     "target tokens -- most of the gradient on generic chat.")
ap.add_argument("--trajectory", type=int, default=0,
                help="agent steps to keep (0 = all). Coverage of every tool "
                     "is preserved before the remainder is sampled.")
ap.add_argument("--tool-mode", default="names_desc",
                choices=("none", "names", "names_desc", "full"),
                help="what trajectory prompts declare. names_desc matches "
                     "Brittain Code's progressive disclosure at 32k context.")
args = ap.parse_args()


def load(name):
    path = "%s/%s" % (args.data, name)
    return [json.loads(line) for line in open(path, encoding="utf-8")]


BAKED = json.load(open("%s/tool_defs.json" % args.data, encoding="utf-8"))
if isinstance(BAKED, dict):
    BAKED = BAKED.get("tools") or []


def shrink(tool):
    """A tool as the app declares it under progressive disclosure.

    The chat template renders each entry with `tool | tojson`, so dropping the
    parameters here is exactly what the served prompt looks like. Keeping the
    description matters: the model has to pick from these, and a bare name
    carries almost nothing for the tools it has not memorised.
    """
    function = tool.get("function") or tool
    if args.tool_mode == "full":
        return tool
    entry = {"name": function.get("name")}
    if args.tool_mode == "names_desc":
        description = (function.get("description") or "").strip()
        # One sentence. The full descriptions run to several lines and would
        # put most of the schema back.
        entry["description"] = description.split(". ")[0].rstrip(".") + "." if description else ""
    return {"type": "function", "function": entry}


def tools_for(row):
    """The baked tools, plus whatever MCP tools that session had."""
    if args.tool_mode == "none":
        return None
    declared = [shrink(tool) for tool in BAKED]
    for name in row.get("unbaked_tools") or []:
        # Only the name was harvested for these, so only the name is declared.
        declared.append({"type": "function", "function": {"name": name}})
    return declared


def clean_message(message):
    """Only what the model should see. Display keys were stripped at harvest,
    but tool results carry ids that mean nothing outside the original run."""
    out = {"role": message.get("role"), "content": message.get("content") or ""}
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    if message.get("tool_call_id"):
        out["tool_call_id"] = message["tool_call_id"]
    return out


rows = []

# --- trajectories: the agent steps, tools stripped from the prompt ----------
for row in load("trajectories_train.jsonl"):
    target = row.get("target") or {}
    if not (target.get("content") or "").strip() and not target.get("tool_calls"):
        continue                      # the loop's stall path, not an output
    rows.append({
        "kind": "trajectory",
        "tools": tools_for(row),
        "messages": [clean_message(m) for m in row.get("context") or []],
        "target": {
            "content": target.get("content") or "",
            "thinking": target.get("thinking") or "",
            "tool_calls": target.get("tool_calls") or [],
        },
        "meta": {
            "chat_id": row.get("chat_id"),
            "mode": row.get("mode"),
            "has_trace": bool(row.get("has_trace")),
            # Recorded so a tools-present variant can be built later without
            # re-harvesting, and so MCP calls can be found again.
            "unbaked_tools": row.get("unbaked_tools") or [],
        },
    })

# --- size the trajectory slice, keeping one example of every tool ----------
if args.trajectory and args.trajectory < len(rows):
    def target_tool(row):
        calls = row["target"]["tool_calls"] or []
        if not calls:
            return None
        fn = calls[0].get("function") or calls[0]
        return fn.get("name")

    ordered = sorted(rows, key=lambda r: hashlib.sha256(
        json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    keep, seen = [], set()
    for row in ordered:                      # one per tool first
        name = target_tool(row)
        if name and name not in seen:
            seen.add(name)
            keep.append(row)
    covered = len(keep)
    for row in ordered:                      # then fill by hash order
        if len(keep) >= args.trajectory:
            break
        if row not in keep:
            keep.append(row)
    print("trajectory: %d of %d kept, covering %d distinct tools"
          % (len(keep), len(rows), covered))
    if covered < 40:
        raise SystemExit("only %d tools appear in the kept steps; a tool with "
                         "no example cannot be baked in" % covered)
    rows = keep

# --- identity and general: plain chat, target is the last assistant turn ----
def subsample(items, keep):
    """Deterministic subset: sort by a hash of the row and take the first N.

    Not random.sample -- this must give the same rows for the same file
    regardless of seed or read order, so two mixes built a week apart differ
    only where the data differs.
    """
    if keep <= 0 or keep >= len(items):
        return items
    ordered = sorted(items, key=lambda row: hashlib.sha256(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    return ordered[:keep]


for name, kind in (("identity_sft.jsonl", "identity"),
                   ("general_sft.jsonl", "general"),
                   # Tool failures that will not resolve, and writing tasks that
                   # need no tool. Same shape as the sets above: the context
                   # carries the assistant/tool turns, the last turn is the
                   # target. See build_restraint_sft.py for why it exists.
                   ("restraint_sft.jsonl", "restraint"),
                   # Python <-> BrittainScript, sampled from the parallel
                   # corpus with the eval's hashes excluded. See
                   # build_bs_sft.py: the base knows the name from PyPI
                   # metadata and invents the language.
                   ("bs_sft.jsonl", "brittainscript")):
    source = load(name)
    if kind == "general":
        # A bare label is not a conversational answer. These are multiple
        # choice keys and classification outputs from the benchmark subsets;
        # trained as chat they teach one-word replies to everything.
        def answers_in_words(row):
            messages = row.get("messages") or []
            if not messages:
                return False
            reply = (messages[-1].get("content") or "").strip()
            return len(reply.split()) >= 4

        before = len(source)
        source = [row for row in source if answers_in_words(row)]
        print("general: dropped %d bare-label answers of %d"
              % (before - len(source), before))
        source = subsample(source, args.general)
    for row in source:
        messages = row.get("messages") or []
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            continue
        rows.append({
            "kind": kind,
            "messages": [clean_message(m) for m in messages[:-1]],
            "target": {
                "content": messages[-1].get("content") or "",
                "thinking": "",
                "tool_calls": [],
            },
            "meta": {k: row.get(k) for k in ("kind", "system_mode", "bucket", "source")
                     if row.get(k)},
        })

# --- remove the machine owner before anything is written -------------------
# A run answered with the wrong surname, read out of the owner home path in
# 29 trained targets. See scrub.py.
rows = [scrub(row) for row in rows]
left = residue(rows)
if left:
    raise SystemExit("scrubbing missed: %s" % left)
print("scrubbed: no owner name, home path or personal address remains")

# Deterministic order. Sorting by a hash of the content rather than shuffling a
# list means the order does not depend on the order the files were read, and is
# reproducible without carrying a seed around.
rows.sort(key=lambda r: hashlib.sha256(
    json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
random.Random(args.seed).shuffle(rows)

with open(args.out, "w", encoding="utf-8") as fh:
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

# --- report what was actually built, not what was intended ------------------
by_kind = collections.Counter(r["kind"] for r in rows)
chars = collections.Counter()
for row in rows:
    size = sum(len(str(m.get("content") or "")) for m in row["messages"])
    size += len(row["target"]["content"]) + len(json.dumps(row["target"]["tool_calls"]))
    chars[row["kind"]] += min(size, args.window * 4)      # rough window cap

print("=" * 72)
print("%-14s %-10s %-12s %s" % ("kind", "examples", "by example", "approx by token*"))
print("=" * 72)
total_chars = sum(chars.values())
for kind in ("trajectory", "identity", "general", "restraint",
             "brittainscript"):
    print("%-14s %-10d %-12.0f%% %.0f%%"
          % (kind, by_kind[kind], 100.0 * by_kind[kind] / len(rows),
             100.0 * chars[kind] / total_chars))
print("=" * 72)
print("total: %d examples -> %s" % (len(rows), args.out))
declared = next((r["tools"] for r in rows if r.get("tools")), None)
print("tool-mode: %s (%s)"
      % (args.tool_mode,
         "%d tools declared on trajectory prompts" % len(declared)
         if declared else "trajectory prompts declare no tools"))
print("* chars/4 with the window cap applied; the trainer reports exact tokens.")

with_calls = sum(1 for r in rows if r["target"]["tool_calls"])
with_trace = sum(1 for r in rows if r["meta"].get("has_trace"))
print("\ntargets that call a tool : %d" % with_calls)
print("targets carrying a trace : %d  (the trainer decides whether to train them)"
      % with_trace)

restraint = [r for r in rows if r["kind"] == "restraint"]
if restraint:
    called = sum(1 for r in restraint if r["target"]["tool_calls"])
    print("\nrestraint targets that call a tool: %d of %d (must be 0)"
          % (called, len(restraint)))
    if called:
        raise SystemExit("a restraint target calls a tool; that trains the "
                         "opposite of what the set is for")
