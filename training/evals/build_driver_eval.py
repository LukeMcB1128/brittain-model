"""Build the held-out driver eval from harvested trajectories.

    python3 build_driver_eval.py --in trajectories.jsonl --defs tool_defs.json \
        --out eval_driver.jsonl

SPLIT BY CHAT, NOT BY EXAMPLE
Every assistant turn in a conversation shares that conversation's history, so a
per-example split puts near-identical contexts on both sides and reports a
score that is mostly memorisation. Splitting on a hash of the chat id keeps a
whole conversation on one side, and pins it there as the corpus grows -- the
same reason filter_bs.py hashes content instead of shuffling.

WINDOW BUDGET
The 55 schemas are 8,082 estimated tokens and get sent on every request until
they are baked, so the context has to leave room for them plus generation
inside the 32k we serve. Windowing to 20,480 leaves ~12k of headroom.

WHAT THE REFERENCE IS AND IS NOT
The target is what a stronger model actually did -- qwen3.6:35b, ornith:35b,
gpt-oss:20b. It is a reference, not ground truth: agreement with it is a
useful signal, but the load-bearing metrics are the objective ones (did a
parseable call come out, is the name real, do the arguments validate), which
need no reference at all.
"""
import argparse
import collections
import hashlib
import json

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", required=True)
ap.add_argument("--defs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--window", type=int, default=20480)
ap.add_argument("--val-frac", type=float, default=0.20)
ap.add_argument("--max-examples", type=int, default=250,
                help="eval runs on the serving stack at ~50 tok/s; keep it affordable")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
defs = json.load(open(args.defs, encoding="utf-8"))
live = {d["function"]["name"] for d in defs}


def render(msg):
    parts = [msg.get("role") or "", msg.get("content") or ""]
    if msg.get("thinking"):
        parts.append(msg["thinking"])
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        parts.append(fn.get("name") or "")
        parts.append(json.dumps(fn.get("arguments") or {}))
    if msg.get("tool_name"):
        parts.append(msg["tool_name"])
    return "\n".join(p for p in parts if p)


def ntok(text):
    return len(tok(text).input_ids) if text else 0


def held_out(chat_id):
    h = hashlib.sha256(str(chat_id).encode()).hexdigest()
    return int(h[:8], 16) % 10_000 < args.val_frac * 10_000


def window(context, budget):
    costs = [ntok(render(m)) for m in context]
    if not context:
        return []
    spend, start = 0, len(context)
    for i in range(len(context) - 1, -1, -1):
        if spend + costs[i] > budget:
            break
        spend += costs[i]
        start = i
    while start < len(context) and context[start].get("role") != "user":
        start += 1
    return context[start:]


def mcp_defs_for(context):
    """Reconstruct the MCP tools that were connected during this conversation.

    MCP servers append definitions at runtime, so a context containing
    playwright calls is a context where playwright's schemas WERE sent. Not
    sending them makes the model look like it hallucinates names when it is
    really imitating tools it can see were available -- an artefact of the
    eval, not a property of the model.

    The real schemas are gone, so these are reconstructed from observed usage:
    the name, and the union of argument keys ever passed to it.
    """
    observed = collections.defaultdict(set)
    for m in context:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            name = fn.get("name")
            if not name or name in live:
                continue
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except ValueError:
                    a = {}
            if isinstance(a, dict):
                observed[name].update(a)
            else:
                observed[name]
    return [{
        "type": "function",
        "function": {
            "name": name,
            "description": "Tool provided by a connected MCP server.",
            "parameters": {
                "type": "object",
                "properties": {k: {} for k in sorted(keys)},
            },
        },
    } for name, keys in sorted(observed.items())]


rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
kept, seen_chats, stats = [], set(), collections.Counter()

for r in rows:
    if not held_out(r["chat_id"]):
        stats["train side"] += 1
        continue
    calls = r["target"].get("tool_calls") or []
    if not calls:
        # Text-only turns are a real behaviour (the loop ends on one) but they
        # are scored differently; keep the first pass focused on tool calls.
        stats["held out, text-only"] += 1
        continue
    names = [(c.get("function") or {}).get("name") for c in calls]
    if any(n not in live for n in names):
        # We do not have the MCP servers' schemas, so we cannot send them --
        # and a model that has not been given playwright's definitions cannot
        # fairly be expected to call playwright. Generalising to unseen tools
        # is a real capability, but it needs its own eval with real schemas.
        stats["held out, references MCP"] += 1
        continue
    ctx = window(r["context"], args.window)
    if not ctx:
        stats["no usable context"] += 1
        continue
    seen_chats.add(r["chat_id"])
    kept.append({
        "chat_id": r["chat_id"],
        "mode": r["mode"],
        "think": r["think"],
        "source_model": r["source_model"],
        "context": ctx,
        "reference": {
            "tool_names": [(c.get("function") or {}).get("name") for c in calls],
            "arguments": [(c.get("function") or {}).get("arguments") for c in calls],
        },
        "extra_tools": mcp_defs_for(ctx),
        "reference_unbaked": sorted(
            {n for n in ((c.get("function") or {}).get("name") for c in calls)
             if n not in live}),
    })
    stats["held out, tool-calling"] += 1

# Even coverage beats sheer count: cap per tool so read_file does not dominate.
by_tool = collections.defaultdict(list)
for ex in kept:
    by_tool[ex["reference"]["tool_names"][0]].append(ex)
selected, cap = [], max(3, args.max_examples // max(1, len(by_tool)))
for name in sorted(by_tool):
    selected.extend(by_tool[name][:cap])
selected = selected[:args.max_examples]

with open(args.out, "w", encoding="utf-8") as f:
    for ex in selected:
        f.write(json.dumps(ex) + "\n")

print("rows in            : %d" % len(rows))
for why, n in stats.most_common():
    print("  %-24s %d" % (why, n))
print("\nheld-out chats     : %d" % len(seen_chats))
print("eval examples      : %d  (cap %d per tool)" % (len(selected), cap))
print("distinct first tool: %d" % len({e["reference"]["tool_names"][0] for e in selected}))
counts = collections.Counter(e["reference"]["tool_names"][0] for e in selected)
print("\nreference tool distribution:")
for name, n in counts.most_common():
    mark = "" if name in live else "  (not in the 55 -- MCP)"
    print("   %-34s %3d%s" % (name, n, mark))
print("\nwrote %s" % args.out)
