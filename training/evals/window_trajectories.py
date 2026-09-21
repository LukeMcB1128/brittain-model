"""Fit harvested trajectories into the window we actually serve.

    python3 window_trajectories.py --in trajectories.jsonl --window 32768
    python3 window_trajectories.py --in ... --window 4096 --stats

WHY THIS IS A SEPARATE PASS
Harvesting walks 20 MB of chat JSON; windowing is a tokenizer loop over the
result. Baking a window size into the harvester would mean re-walking the
corpus every time the serving context changed -- and it has already changed
twice this week (16k, then 32k). trajectories.jsonl stays the raw replay and
this produces the training file from it.

WHY THE RAW REPLAY IS THE WRONG THING TO TRAIN ON
The stored conversation is the whole history. The model never saw it: these
chats ran with requestedContextCap 131072 and auto-compaction at 0.70 of the
window, so by the time history was long the model was reading a compacted
version. Replaying it whole produces contexts up to 195,746 tokens -- a
distribution that cannot occur at serving time, where brittain4 has 32k.

WHAT THIS DOES INSTEAD
Keeps the most recent messages that fit, snapped to a user boundary, and always
retains the first user message. The app's real compaction replaces dropped
history with a GOAL/CONSTRAINTS/DECISIONS/STATE/NEXT summary; keeping the
opening request is a cheap stand-in for GOAL, and without it a truncated
example asks the model to continue work whose purpose was cut away.

Examples whose TARGET alone exceeds the window are dropped -- nothing can be
done with them at this size.
"""
import argparse
import collections
import json

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", required=True)
ap.add_argument("--out", default=None, help="default: <in>.w<window>.jsonl")
ap.add_argument("--window", type=int, default=32768)
ap.add_argument("--reserve", type=int, default=512,
                help="headroom for the frame and generation")
ap.add_argument("--stats", action="store_true")
args = ap.parse_args()

OUT = args.out or args.src.replace(".jsonl", "") + ".w%d.jsonl" % args.window
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")


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


def window(context, budget):
    """Most recent messages fitting in budget, snapped to a user boundary.

    Snapping matters because a tool result whose call was cut is unreadable --
    the model would see an answer with no question. The app's own compaction
    keeps a verbatim tail starting at role:'user' for the same reason.

    The opening request is prepended as a stand-in for the compaction summary's
    GOAL section, but only when it is small enough to be worth the room. An
    early version prepended it unconditionally, and a single 42,595-token paste
    then appeared in every window including 4096.
    """
    costs = [ntok(render(m)) for m in context]
    if not context:
        return [], 0, False

    head, head_cost = [], 0
    if context[0].get("role") == "user" and costs[0] <= budget // 4:
        head, head_cost = [context[0]], costs[0]

    spend, start = 0, len(context)
    for i in range(len(context) - 1, 0, -1):
        if head_cost + spend + costs[i] > budget:
            break
        spend += costs[i]
        start = i

    while start < len(context) and context[start].get("role") != "user":
        spend -= costs[start]
        start += 1

    kept = context[start:]
    truncated = start > 0
    if truncated and head:
        kept = head + kept
        spend += head_cost
    return kept, spend, truncated


rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
out = None if args.stats else open(OUT, "w", encoding="utf-8")

stats = collections.Counter()
lengths = []
for r in rows:
    tgt = render({"role": "assistant", **r["target"]})
    tgt_n = ntok(tgt)
    budget = args.window - args.reserve - tgt_n
    if budget <= 0:
        stats["target alone exceeds window"] += 1
        continue

    kept, spend, truncated = window(r["context"], budget)
    stats["truncated" if truncated else "fit whole"] += 1
    total = spend + tgt_n
    if total > args.window:
        stats["still over window after trimming"] += 1
        continue
    lengths.append(total)
    if out:
        dropped_msgs = len(r["context"]) - len(kept)
        r["context"] = kept
        r["windowed"] = {"window": args.window, "tokens": total,
                         "truncated": truncated, "dropped_messages": dropped_msgs}
        out.write(json.dumps(r) + "\n")
if out:
    out.close()

lengths.sort()
print("examples in : %d" % len(rows))
print("examples out: %d" % len(lengths))
for why, n in stats.most_common():
    print("  %-28s %d" % (why, n))
if lengths:
    def pct(p):
        return lengths[min(len(lengths) - 1, int(len(lengths) * p))]
    print("\ntokens per example at window %d:" % args.window)
    for p in (0.5, 0.9, 0.99):
        print("   p%-3d %7d" % (p * 100, pct(p)))
    print("   max  %7d" % lengths[-1])
    print("   total %s tokens" % f"{sum(lengths):,}")
if not args.stats:
    print("\nwrote %s" % OUT)
